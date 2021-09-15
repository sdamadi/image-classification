def init_writer(args, scenario, distributed):

    if args.local_rank == 0:
        # creating all directories to save models, numpys, figures, and terminal logs
        dirs = Directories(args, scenario.curr_scen_name)
        sys.stdout = Logger(dirs.logpath, args.logterminal)
        print('\n========> The following is the setup for this run:\n')
        print(f'{scenario.curr_scen_name}\n')
        for arg in vars(args):
            print(f'{arg}: {getattr(args, arg)}')
        # writing the terminal output into a file
        logdir = f'./runs/{args.dataname}/{args.arch}/{scenario.time}'
        if not os.path.exists(logdir): os.makedirs(logdir)
        writer = SummaryWriter(log_dir=logdir, comment = f'_{scenario.curr_scen_name}')
        print(f"\n=> Global rank of the current node is {torch.distributed.get_rank() if distributed.args.world_size>1 else 0}"\
            f" and the process id is {os.getpid()}."\
            f"\n=> There are {distributed.args.world_size} process(es) runing on GPU(s)."\
            f"\n=> Visible GPU(s) are {args.gpu_idx} for running {distributed.args.world_size} process(es)."\
            f"\n=> Execute `nvidia-smi` on a differnt terminal to see used GPUs."\
            f"\n=> GPU {args.gpu_idx[args.local_rank]} whose id is {args.local_rank} is being used for training of the current process.\n")
    
    return writer
