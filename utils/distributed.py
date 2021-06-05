import os, sys
import torch
from torch.utils.tensorboard import SummaryWriter
from utils.directories import Directories
from utils.logger import Logger


class Distributed(object):
    def __init__(self, args):
        # It is initialized so as to be `False`, however, it may change.
        self.distributed = False
        self.args = args
        self.dist_elements()
    
    def dist_elements(self):

        self.args.distributed = False
        #By passing `-m torch.distributed.launch --nproc_per_node=4` one sets the `WORLD SIZE VALUE`
        # to 4 and main function is spawned 4 times where each one will be on one of 4 GPUs
        if 'WORLD_SIZE' in os.environ:
            self.args.distributed = int(os.environ['WORLD_SIZE']) > 1
        
        if self.args.distributed:
        
            
            
            
            # `args.gpu_idx` is a list of indices of gpus each on which
            # each process will be run and would be the same for every node 
            
            if self.args.gpu_idx == 'None':
                self.args.gpu_idx = [f'{i}' for i in range(torch.cuda.device_count())]
                s = ','.join(self.args.gpu_idx)
                os.environ['CUDA_VISIBLE_DEVICES'] = s     
            else:
                #creating a string for `CUDA_VISIBLE_DEVICES`
                self.args.gpu_idx = [f'{i}' for i in self.args.gpu_idx]
                s = ','.join(self.args.gpu_idx)
                os.environ['CUDA_VISIBLE_DEVICES'] = s
            
            # this is the total # of GPUs across all nodes
            # if using 2 nodes with 4 GPUs each, world size is 8
            torch.distributed.init_process_group(backend='nccl', init_method='env://')
            self.args.world_size = torch.distributed.get_world_size()
            self.args.local_rank = torch.distributed.get_rank()
            self.cuda_id = self.args.local_rank
            torch.cuda.set_device(self.cuda_id)              
                    
        else:
            self.args.world_size = 1
            
            if self.args.gpu_idx == 'None':
                self.args.gpu_idx = [f'{i}' for i in range(torch.cuda.device_count())]
                s = ','.join(self.args.gpu_idx)
                os.environ['CUDA_VISIBLE_DEVICES'] = s
            else:
                self.args.gpu_idx = [f'{i}' for i in self.args.gpu_idx]
                s = ','.join(self.args.gpu_idx)
                os.environ['CUDA_VISIBLE_DEVICES'] = s
            
            self.cuda_id = self.args.local_rank
            torch.cuda.set_device(self.cuda_id)
                    
        
        assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."

def init_desc(args, scenario, distributed):

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

        