import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP
from utils import *

def main():

    args = parse()
    
    cudnn.benchmark = True

    #runing n process(es) on n GPUs when `--nproc_per_node>1`
    distributed = Distributed(args)
    args.world_size, args.distributed = distributed.args.world_size, distributed.args.distributed
    args.gpu_ids, cuda_id = distributed.args.gpu_ids, distributed.cuda_id
    
    #creating the name of current process
    scenario = Scenario(args)
     
    # Only first process (`local_rank = 0`) can log, create directories, 
    # save variables, and write to the TensorBoard 
    if args.local_rank == 0:
        writer = init_writer(args, scenario, distributed)

    exit()

    # ===== Dataset ===== #

    train_dataset, validation_dataset = get_data(args)
    
    train_loader, validation_loader, train_sampler = loaders(train_dataset,
                                                        validation_dataset, args)

    # ===== Model creation ===== # 
    model = Archs(args).model.cuda(cuda_id)

    # ===== Model initialization ===== # 
    network_init(model, args)

 
    if args.distributed:
        model = DDP(model, device_ids = [cuda_id], output_device = cuda_id)


    # ===== Optimizer ===== # 
    optimizer = SFO(model, args)

    # ===== Defining loss function ===== #
    if args.label_smoothing == 0:
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

   
    if args.local_rank == 0:
        outputwriter.net_params_init()

    for epoch in range(args.initial_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        #train for one epoch
        traval.train(epoch)        
        traval.validation(epoch, report=True)
    
    traval.stage_quantities()

    # saves params
    if args.local_rank == 0:
        outputwriter.net_params()
            
    if args.local_rank == 0:
        writer.close()
        traval.close()
        outputwriter.close()
    return

if __name__ == '__main__':
    main()


    
    
    


