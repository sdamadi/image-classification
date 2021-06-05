import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP
import copy

from utils import *

def main():

    args = parse()
    cudnn.benchmark = True

    #runing n processes on n GPUs when `self.args.distributed = True`
    distributed = Distributed(args)
    args.world_size, args.distributed = distributed.args.world_size, distributed.args.distributed
    args.gpu_idx, cuda_id = distributed.args.gpu_idx, distributed.cuda_id

    #creating the name of current process
    scenario = Scenario(args)

    # Only first (`local_rank = 0`) process can log, create directories, 
    # save variables, and write to tensorboard 
    if args.local_rank == 0:
        writer = init_desc(args, scenario, distributed) 

    # ===== Dataset ===== #

    train_dataset, validation_dataset = get_data(args)
    
    train_loader, validation_loader, train_sampler = loaders(train_dataset,
                                                        validation_dataset, args)

    # ===== Model creation ===== # 
    net = Archs(args).model.cuda(cuda_id)
    if not (args.prepruned_model or args.quantize_prepruned):
        network_init(net, args.init_policy, args.init_kaiming_mode, args.init_kaiming_nonlinearity, args.init_bias)
    if args.prepruned_model or args.quantize_prepruned:
        net = get_sparse_init(net, args.dataname, args.arch, args.prepruned_model, 
                                args.prepruned_scen, 
                                args.nonpruned_percent, args.mask_stage,
                                args.local_rank, args.quantize_prepruned, args.quantize_bias) 
    if args.distributed:
        model = DDP(net, device_ids = [cuda_id], output_device = cuda_id)
    else:
        model = copy.deepcopy(net)
    if args.prepruned_model or args.distributed: del net

    # ===== Optimizer ===== # 
    optimizer = SFO(model, args)

    # ===== Defining loss function ===== #
    if args.label_smoothing is None:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = LabelSmoothing(smoothing=args.label_smoothing)

    # ===== Automatic mixed precision ===== # 
    scaler = torch.cuda.amp.GradScaler()


    # ===== Pruning ===== # 
    poda = Poda(model, args)

    # printing the status of parameters including layers, elements of layers, etc
    poda.params_status(stage = 0, report_layer = False if args.local_rank != 0 else True,
                         report_network = False if args.local_rank != 0 else True)


    # ===== Train and validation ===== # 
    traval = TraVal(model, train_loader, optimizer,
                 criterion, scaler,
                 args, validation_loader,
                 writer = writer if args.local_rank == 0 else None,
                 mask = None if (not args.prepruned_model) else poda.mask,
                 eta = poda.eta,
                 curr_scen_name = scenario.curr_scen_name if args.local_rank == 0 else None)

    # ===== Saving models and variables ===== #
    if args.local_rank == 0:
        outputwriter = Outputwriter(model, poda, traval, writer, args,
                                    scenario.curr_scen_name)

    p = 0
    # find the value of pi
    s = 0
    torch.pi = torch.acos(-torch.ones(1)).item()
    for stage in range(0, 1 if args.stages == 0 else args.stages + args.initial_stage):

        if stage == 0 and args.local_rank == 0:
            outputwriter.net_params_init()

        for epoch in range(0, args.epochs):
            if args.distributed :#and args.dataname == 'imagenet'
                train_sampler.set_epoch(epoch)

            #train for one epoch
            traval.eta = poda.eta
            traval.train(epoch, stage)        
            traval.validation(epoch, stage, report=True)
        
        traval.best_values(stage)
        traval.stage_quantities()

        # saves pruned params
        if args.local_rank == 0:
            outputwriter.net_params(stage)
        
        if args.pruning_strategy in {'asni', 'lottery'} and not args.prepruned_model: 
            if stage >= args.initial_stage:
                if args.stages != 0:
                    if args.pruning_strategy == 'lottery':
                        p = args.percent
                    elif args.pruning_strategy == 'asni': 

                        if args.asni_mode == 'sine':
                            
                            alpha = torch.tensor(torch.pi*stage/(args.asni_sin_scale*(args.stages + args.initial_stage)), 
                                                dtype = torch.float)
                            perc_mode = args.asni_sin_mag*torch.sin(alpha)**args.asni_sin_exponent 
                            for i in range(10): print(f'sine mode and perc_mode{perc_mode:4.2f}')
                        elif args.asni_mode == 'sigmoid':
                            # trans = (args.stages if args.initial_stage == 0 else args.stages - args.initial_stage)*args.asni_sigmoid_trans
                            
                            if stage < args.asni_rest_stage:
                                trans = args.asni_sigmoid_trans_1*(args.asni_rest_stage if args.initial_stage == 0 else args.asni_rest_stage-args.initial_stage)
                                scale = args.asni_sigmoid_scale_1
                                argument = (stage-trans)/scale
                                perc_mode = args.asni_sigmoid_mag_1*torch.sigmoid(torch.tensor(argument))
                                y0 = perc_mode
                            elif  args.asni_rest_stage <= stage < args.stages:
                                trans = args.asni_sigmoid_trans_2*(args.stages - args.asni_rest_stage)
                                scale = args.asni_sigmoid_scale_2
                                argument = ( (stage-args.asni_rest_stage)- trans)/scale
                                perc_mode= y0 + args.asni_sigmoid_mag_2*torch.sigmoid(torch.tensor(argument))
                            
                            
                            # scale = args.asni_sigmoid_scale
                            # x = torch.tensor((stage-trans)/scale)
                            # perc_mode = args.asni_sigmoid_mag*torch.sigmoid(x)

                        _p = 100*(perc_mode-s)/(100-s)
                        s = perc_mode
                        p = _p if _p > 0.1 else 0

                    if p != 0 and args.asni_perc_max > perc_mode:
                        poda.subnetwork_picker(model, p, args.local_prune, 
                                                args.prune_bn, args.prune_bias)
                        traval.mask = poda.mask
                        traval.pruned_percent = poda.pruned
            
            if args.local_rank == 0: 
                outputwriter.threshold(stage)
        
        poda.params_status(stage+1 , report_layer = False if args.local_rank != 0 else True,
                        report_network = False if args.local_rank != 0 else True)
        
            
    if args.local_rank == 0:
        writer.close()
        traval.close()
        outputwriter.close()
    return
   

    # to do
        # - follow the previous main file 

if __name__ == '__main__':
    main()


    
    
    


