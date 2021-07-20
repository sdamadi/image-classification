import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP
import copy

from utils import *

def main():

    args = parse()
    cudnn.benchmark = True

    #runing n processes on n GPUs when `--nproc_per_node>1`
    distributed = Distributed(args)
    args.world_size, args.distributed = distributed.args.world_size, distributed.args.distributed
    args.gpu_idx, cuda_id = distributed.args.gpu_idx, distributed.cuda_id

    #creating the name of current process
    scenario = Scenario(args)

    # Only first process (`local_rank = 0`) can log, create directories, 
    # save variables, and write to the TensorBoard 
    if args.local_rank == 0:
        writer = init_desc(args, scenario, distributed) 

    # ===== Dataset ===== #

    train_dataset, validation_dataset = get_data(args)
    
    train_loader, validation_loader, train_sampler = loaders(train_dataset,
                                                        validation_dataset, args)

    # ===== Model creation ===== # 
    net = Archs(args).model.cuda(cuda_id)
    network_init(net, args.init_policy, args.init_kaiming_mode, args.init_kaiming_nonlinearity, args.init_bias)
 
    if args.distributed:
        model = DDP(net, device_ids = [cuda_id], output_device = cuda_id)
    else:
        model = copy.deepcopy(net)

    # ===== Optimizer ===== # 
    optimizer = SFO(model, args)

    # ===== Defining loss function ===== #
    if args.label_smoothing is None:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = LabelSmoothing(smoothing=args.label_smoothing)

    # ===== Automatic mixed precision ===== # 
    scaler = torch.cuda.amp.GradScaler()

    # ===== Train and validation ===== # 
    traval = TraVal(model, train_loader, optimizer,
                 criterion, scaler,
                 args, validation_loader,
                 writer = writer if args.local_rank == 0 else None,
                 curr_scen_name = scenario.curr_scen_name if args.local_rank == 0 else None)

    # ===== Saving models and variables ===== #
    if args.local_rank == 0:
        outputwriter = Outputwriter(model, traval, writer, args,
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
            traval.train(epoch, stage)        
            traval.validation(epoch, stage, report=True)
        
        traval.best_values(stage)
        traval.stage_quantities()

        # saves pruned params
        if args.local_rank == 0:
            outputwriter.net_params(stage)
            
    if args.local_rank == 0:
        writer.close()
        traval.close()
        outputwriter.close()
    return


if __name__ == '__main__':
    main()


    
    
    


