import time, os
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.cuda.amp import autocast

from utils.lr_policy import *


class TraVal(object):

    def __init__(self, model, train_loader,
                    optimizer, criterion, scaler,
                    args, validation_loader,
                    writer, mask, eta, curr_scen_name):

        self.model = model
        self.mask = mask
        self.args = args
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.writer = writer
        self.scaler = scaler
        self.eta = eta
        self.pruned_percent = round(float(100*(1-self.eta)))
        self.best_pruned = round(float(100*(1-self.eta)))
        self.loss_vector = list()
        self.accuracy_vector = list()
        self.train_loss = 0
        self.total_loss = 0
        self.train_acc = 0
        self.step = 0

        self.grad_norm = 0
        self.delta_dot_grad = 0
        self.delta2 = 0

        self.GradNorm = []
        self.TrainingLoss, self.TrainingTop1, self.TrainingTop5 = ( [] for i in range(3) )
        self.TestLoss, self.TestTop1, self.TestTop5 = ( [] for i in range(3) )
        self.best_loss_test, self.best_top1_test  = np.inf, 0
        self.Best_Stage, self.Best_Loss_Test, self.Best_Top1_Test, self.Best_Percent = ([] for i in range(4) )
        self.golden = False
        self.golden_stage = 0
        self.global_threshold = 0
        self.Pruning_Percent = [round((1-self.eta.item())*100,2)]
        self.Learning_Rate = [self.args.lr]
        self.curr_scen_name = curr_scen_name
        if args.local_rank == 0:
            self.scen_name, self.scen_time = self.time_remover(self.curr_scen_name)
        self.lr_policy = LR(self.args.lr_policy, self.args.lr, self.args.warmup_length,
                             self.args.epochs, self.args.initial_stage, self.args.stages,
                             self.args.lowest_lr, self.args.scale_coslr, self.args.exp_coslr,
                             self.args.normal_exp_scale,
                             self.args.lr_gamma, self.args.lr_steps)

        self.lr = self.args.lr


    def train(self, epoch, stage):
        self.epoch = epoch
        self.epoch_length = len(self.train_loader)

        self.batch_time_tr = AverageMeter()
        self.losses_tr = AverageMeter()
        self.top1_tr = AverageMeter()
        self.top5_tr = AverageMeter()

        # switch to train mode
        self.model.train()
        self.end_tr = time.time()

        prefetcher = DataPrefetcher(self.train_loader, self.args.dataname)
        input, target = prefetcher.next()
        i = 0
        while input is not None:
        # while i<=20:
            i += 1
            # self.adjust_learning_rate(i, stage)
            self.lr = self.lr_policy.apply_lr(self.epoch, stage)
            self.assign_learning_rate(self.lr)

            # compute output
            with autocast():
                output = self.model(input)
                loss = self.criterion(output, target)            

            # compute gradient and do First Order Update(SGD, SGD+M, Adam) step
            self.optimizer.zero_grad()

            # Mixed-precision training requires that the loss is scaled in order
            # to prevent the gradients from underflow

            self.scaler.scale(loss).backward()

            total_norm = 0
            grad_k = list() 
            for p in self.model.parameters():
                param_norm = p.grad.data.norm(2)
                grad_k.append(p.grad.data.clone())
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            self.grad_norm = total_norm

            # `or self.args.prepruned_model` accounts for the case of training a sparse network  
            if self.args.pruning_strategy in {'asni', 'lottery', 'quantize', 'prepruned'}: #and 
                if self.args.percent == 0 and self.args.pruning_strategy == 'lottery' and not self.args.prepruned_model:
                    self.mask = None
            else:
                self.mask = None
            
            self.scaler.step(self.optimizer, self.mask)
            self.scaler.update()

            self.step += 1

            # Measure accuracy
            self.prec1_tr, self.prec5_tr = self.accuracy(output.data, target, topk=(1, 5))

            # Average across all global processes for logging
            if self.args.distributed:
                self.reduced_loss_tr = self.reduce_tensor(loss.data)
                self.prec1_tr = self.reduce_tensor(self.prec1_tr)
                self.prec5_tr = self.reduce_tensor(self.prec5_tr)
            else:
                self.reduced_loss_tr = loss.data

            # to_python_float incurs a host <-> device sync
            self.losses_tr.update(self.reduced_loss_tr.data.item(), input.size(0))
            self.top1_tr.update(self.prec1_tr.data.item(), input.size(0))
            self.top5_tr.update(self.prec5_tr.data.item(), input.size(0))

            
            self.write_net_values(stage, train=True)
            
            # if torch.distributed.get_rank() == 0:            
            #     self.writer.add_scalar('Loss/train', self.reduced_loss_tr.item(), self.step)
            #     self.writer.add_scalar(f'Gradient/Norm', self.grad_norm, self.step)
            #     self.writer.add_scalar(f'Top1/train', self.prec1_tr.data.item(), self.step)
            #     self.writer.add_scalar(f'Top1/lr', self.lr, self.step)
            #     self.writer.add_scalar(f'Top5/train', self.prec5_tr.data.item(), self.step)
            #     self.writer.add_scalar(f'Top5/lr', self.lr, self.step)


                # self.writer.add_scalar(f'Gradient/Norm', self.traval.grad_norm, self.traval.step)

            torch.cuda.synchronize()
            self.batch_time_tr.update((time.time() - self.end_tr) ) #self.args.print_freq_tr
            self.end_tr = time.time()

            # Every print_freq iterations, check the loss, accuracy, and speed.
            # For best performance, it doesn't make sense to print these metrics every
            # iteration, since they incur an allreduce and some host<->device syncs.

            if i % self.args.print_freq_tr == 0:
                if self.args.local_rank == 0:
                    self.curr_throughput_tr = self.args.world_size*self.args.batch_size/self.batch_time_tr.val
                    self.avg_throughput_tr = self.args.world_size*self.args.batch_size/self.batch_time_tr.avg
                    
                    if stage>= self.args.initial_stage and self.args.percent !=0 and self.args.pruning_strategy in {'asni', 'lottery'}:
                        prune_stg = stage - self.args.initial_stage
                    else:
                        prune_stg = 0
                    
                    if self.args.percent !=0 and self.args.pruning_strategy in {'asni', 'lottery'}:
                        prune_stgs = self.args.stages - 1
                    else:
                        prune_stgs = 0

                    print(f'Stg: [{stage:2}/{self.args.stages+self.args.initial_stage:2}] | '\
                        f'Pruning Stg: [{prune_stg:2}/'\
                        f'{prune_stgs:2}] | '\
                        f'Zero params: {100*(1-self.eta):4.2f} %\n'\
                        f'Num of GPUs:{self.args.world_size:2} | '\
                        f'Epoch: [{self.epoch+1:2}/{self.args.epochs:2}] | '\
                        f'[{i:4}/{len(self.train_loader):4}] | '\
                        f'Time(avg): {self.args.print_freq_tr*self.batch_time_tr.avg:4.2f} | '\
                        f'Speed: (pics/sec): {self.avg_throughput_tr:5.0f}\n'\
                        f'Learning rate: {self.lr:9.8f} | '\
                        f'Curr loss: {self.losses_tr.val:5.4f} | '\
                        f'Avg loss: {self.losses_tr.avg:5.4f} | '\
                        f'Prec@1(avg) {self.top1_tr.avg:4.2f} % | '\
                        f'Prec@5(avg) {self.top5_tr.avg:4.2f} %\n')

            input, target = prefetcher.next()

    def assign_learning_rate(self, new_lr):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr

    
    def validation(self, epoch, stage, report):
        self.epoch = epoch
 
        self.batch_time_ts = AverageMeter()
        self.losses_ts = AverageMeter()
        self.top1_ts = AverageMeter()
        self.top5_ts = AverageMeter()

        # switch to evaluate mode
        self.model.eval()

        self.end_ts = time.time()

        prefetcher = DataPrefetcher(self.validation_loader, self.args.dataname)
        input, target = prefetcher.next()
        i = 0
        
        while input is not None:
        # while i <= 20:
            i += 1
            # compute output
            with torch.no_grad():
                output = self.model(input)
                loss = self.criterion(output, target)

            # measure accuracy and record loss
            self.prec1_ts, self.prec5_ts = self.accuracy(output.data, target, topk=(1, 5))

            if self.args.distributed:
                self.reduced_loss_ts = self.reduce_tensor(loss.data)
                self.prec1_ts = self.reduce_tensor(self.prec1_ts)
                self.prec5_ts = self.reduce_tensor(self.prec5_ts)
            else:
                self.reduced_loss_ts = loss.data

            self.losses_ts.update((self.reduced_loss_ts.data.item()), input.size(0))
            self.top1_ts.update(self.prec1_ts.data.item(), input.size(0))
            self.top5_ts.update(self.prec5_ts.data.item(), input.size(0))

            # measure elapsed time
            self.batch_time_ts.update((time.time() - self.end_ts)*self.args.world_size)
            self.end_ts = time.time()

            input, target = prefetcher.next()

            

            # TODO:  Change timings to mirror train().
            if i % self.args.print_freq_ts == 0 and report:
                if self.args.local_rank == 0:
                    self.curr_throughput_ts = self.args.world_size * self.args.batch_size / self.batch_time_ts.val
                    self.avg_throughput_ts = self.args.world_size * self.args.batch_size / self.batch_time_ts.avg
                    
                    if stage>= self.args.initial_stage and self.args.percent !=0 and self.args.pruning_strategy in {'asni', 'lottery'}:
                        prune_stg = stage - self.args.initial_stage
                    else:
                        prune_stg = 0
                    
                    if self.args.percent !=0 and self.args.pruning_strategy in {'asni', 'lottery'}:
                        prune_stgs = self.args.stages - 1
                    else:
                        prune_stgs = 0

                    print(
                        f'Validation | Stg: [{stage :2}/{self.args.stages + self.args.initial_stage:2}] | '\
                        f'Pruning Stg: [{prune_stg:2}/{prune_stgs:2}] | '\
                        f'Zero params: {100*(1-self.eta):4.2f} %\n'\
                        f'Epoch: [{self.epoch+1:2}/{self.args.epochs:2}] | '\
                        f'Seen data: [{i:4}/{len(self.validation_loader):4}] | '\
                        f'Time(avg): {self.args.print_freq_ts*self.batch_time_ts.avg:4.2f} | '\
                        f'Speed: (pics/sec): {self.avg_throughput_ts:5.0f}\n'\
                        f'Curr loss: {self.losses_ts.val:5.4f} | '\
                        f'Avg loss: {self.losses_ts.avg:5.4f} | '\
                        f'Prec@1(avg) {self.top1_ts.avg:4.2f} % | '\
                        f'Prec@5(avg) {self.top5_ts.avg:4.2f} %\n')
            
            
            
        self.write_net_values(stage, train=False)        
        
        return self.losses_ts.avg, self.top1_ts.avg, self.top5_ts.avg


    def accuracy(self, output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    
    def reduce_tensor(self, tensor):
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= self.args.world_size
        return rt

    def write_net_values(self, stage, train):
        
        if self.args.local_rank == 0 and train: 
            self.writer.add_scalar('Loss/Training', self.reduced_loss_tr.item(), self.step)
            self.writer.add_scalar(f'Optim/Gradient_Norm', self.grad_norm, self.step)
            self.writer.add_scalar(f'Top1/Training', self.prec1_tr.data.item(), self.step)
            self.writer.add_scalar(f'Optim/lr', self.lr, self.step)
            self.writer.add_scalar(f'Top5/Training', self.prec5_tr.data.item(), self.step)
            
            self.TrainingLoss.append(self.reduced_loss_tr.item())
            self.TrainingTop1.append(self.prec1_tr.data.item())
            self.TrainingTop5.append(self.prec5_tr.data.item())
            self.GradNorm.append(self.grad_norm)

        elif self.args.local_rank == 0 and not train:

            # calculate the average
            self.writer.add_scalar('Loss/Test', self.losses_ts.avg, stage*self.args.epochs + self.epoch + 1)
            self.writer.add_scalar('Top1/Test', self.top1_ts.avg, stage*self.args.epochs + self.epoch + 1)
            self.writer.add_scalar('Top5/Test', self.top5_ts.avg, stage*self.args.epochs + self.epoch + 1)

            # self.writer.add_scalar('Loss/Test', self.reduced_loss_ts.data.item(), stage*self.args.epochs + self.epoch + 1)
            # self.writer.add_scalar('Top1/Test', self.prec1_ts.data.item(), stage*self.args.epochs + self.epoch + 1)
            # self.writer.add_scalar('Top5/Test', self.prec5_ts.data.item(), stage*self.args.epochs + self.epoch + 1)


            self.TestLoss.append( self.losses_ts.avg )
            self.TestTop1.append( self.top1_ts.avg )
            self.TestTop5.append( self.top5_ts.avg )   

    def best_values(self, stage):

        if self.args.local_rank == 0:

            self.golden = False
            if self.args.pruning_strategy in {'asni', 'lottery'} and stage+1 != self.args.initial_stage:

                if self.losses_ts.avg < self.best_loss_test:
                    self.best_loss_test = self.losses_ts.avg 
                    self.golden_stage = stage
                    if self.best_top1_test < self.top1_ts.avg:
                        self.best_top1_test = self.top1_ts.avg
                        self.best_pruned = self.pruned_percent
                        
                    # save the best network with quantized parameters
                    self.golden = True
            
            if self.args.pruning_strategy in {'asni', 'lottery'}: 
                self.writer.add_scalar('Best/stage of pruning', self.golden_stage, stage)
                self.Best_Stage.append(self.golden_stage) 

                self.writer.add_scalar('Best/Loss_Val', self.best_loss_test, stage)
                self.Best_Loss_Test.append(self.best_loss_test)  

                self.writer.add_scalar('Best/Top1', self.best_top1_test, stage)
                self.Best_Top1_Test.append(self.best_top1_test) 

                self.writer.add_scalar('Best/Percent', self.best_pruned, stage)
                self.Best_Percent.append(self.best_pruned) 

    def stage_quantities(self):
       
        self.Pruning_Percent.append(self.pruned_percent) 
        self.Learning_Rate.append(self.lr)

    def close(self):
        root = 'history/variables'

        if self.args.pruning_strategy == 'lottery' and self.args.percent == 0 and not self.args.prepruned_model:
            self.mode = 'trainandtest'
        # elif self.args.percent == 0 and self.args.prepruned_model:
        elif self.args.prepruned_model:
            self.mode = 'trainedsparse'
        elif self.args.pruning_strategy in {'asni', 'lottery', 'str'} and not self.args.prepruned_model:
            self.mode = 'pruned'

        # scen_name, scen_time = self.time_remover(self.curr_scen_name)
        path1 = f'./{root}/{self.args.dataname}/{self.args.arch}/{self.mode}/main'
        self.folder_builder(path = path1, folder_name = self.scen_time)
        path = f'{path1}/{self.scen_time}/'

        file_name = f'TrainingLoss_{self.scen_name}.npy'
        np.save(path + file_name, self.TrainingLoss)

        file_name = f'TrainingTop1_{self.scen_name}.npy'
        np.save(path + file_name, self.TrainingTop1)

        file_name = f'TrainingTop5_{self.scen_name}.npy'
        np.save(path + file_name, self.TrainingTop5)

        file_name = f'GradNorm_{self.scen_name}.npy'
        np.save(path + file_name, self.GradNorm)

        file_name = f'TestLoss_{self.scen_name}.npy'
        np.save(path + file_name, self.TestLoss)

        file_name = f'TestTop1_{self.scen_name}.npy'
        np.save(path + file_name, self.TestTop1)

        file_name = f'TestTop5_{self.scen_name}.npy'
        np.save(path + file_name, self.TestTop5)

        file_name = f'Pruning_Percent_{self.scen_name}.npy'
        np.save(path + file_name, self.Pruning_Percent)

        file_name = f'Learning_Rate_{self.scen_name}.npy'
        np.save(path + file_name, self.Learning_Rate)

        if self.args.pruning_strategy in {'asni', 'lottery'}:    

            file_name = f'Best_Stage_{self.scen_name}.npy'
            np.save(path + file_name, self.Best_Stage)

            file_name = f'Best_Loss_Test_{self.scen_name}.npy'
            np.save(path + file_name, self.Best_Loss_Test)

            file_name = f'Best_Top1_Test_{self.scen_name}.npy'
            np.save(path + file_name, self.Best_Top1_Test)

            file_name = f'Best_Percent_{self.scen_name}.npy'
            np.save(path + file_name, self.Best_Percent)


    def time_remover(self, curr_scen_name):
        scen_name = "_".join(curr_scen_name.split('_')[6:])
        scen_time = "_".join(curr_scen_name.split('_')[:6])

        return scen_name, scen_time

    def folder_builder(self, path, folder_name):
        if not os.path.exists(f'{path}/{folder_name}'):
            os.makedirs(f'{path}/{folder_name}')

            
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        

class DataPrefetcher(object):
    """
    With Amp, it isn't necessary to manually convert data to half.
    """
    def __init__(self, loader, dataname):
        self.loader = iter(loader)
        self.dataname = dataname
        self.stream = torch.cuda.Stream()
        if self.dataname == 'imagenet':
            self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
            self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)

        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        # if record_stream() doesn't work, another option is to make sure device inputs are created
        # on the main stream.
        # self.next_input_gpu = torch.empty_like(self.next_input, device='cuda')
        # self.next_target_gpu = torch.empty_like(self.next_target, device='cuda')
        # Need to make sure the memory allocated for next_* is not still in use by the main stream
        # at the time we start copying to next_*:
        # self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            # more code for the alternative if record_stream() doesn't work:
            # copy_ will record the use of the pinned source tensor in this side stream.
            # self.next_input_gpu.copy_(self.next_input, non_blocking=True)
            # self.next_target_gpu.copy_(self.next_target, non_blocking=True)
            # self.next_input = self.next_input_gpu
            # self.next_target = self.next_target_gpu

            self.next_input = self.next_input.float()
            if self.dataname == 'imagenet':
                self.next_input = self.next_input.sub_(self.mean).div_(self.std)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        if input is not None:
            input.record_stream(torch.cuda.current_stream())
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        self.preload()
        return input, target

        

class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
    