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

            # `args.gpu_ids` is a list of indices of gpus each on which
            # each process will be run and would be the same for every node 
            
            if self.args.gpu_ids == 'None':
                self.args.gpu_ids = [f'{i}' for i in range(torch.cuda.device_count())]
                s = ','.join(self.args.gpu_ids)
                os.environ['CUDA_VISIBLE_DEVICES'] = s     
            else:
                #creating a string for `CUDA_VISIBLE_DEVICES`
                self.args.gpu_ids = [f'{i}' for i in self.args.gpu_ids]
                s = ','.join(self.args.gpu_ids)
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
            
            if self.args.gpu_ids == 'None':
                self.args.gpu_ids = [f'{i}' for i in range(torch.cuda.device_count())]
                s = ','.join(self.args.gpu_ids)
                os.environ['CUDA_VISIBLE_DEVICES'] = s
            else:
                self.args.gpu_ids = [f'{i}' for i in self.args.gpu_ids]
                s = ','.join(self.args.gpu_ids)
                os.environ['CUDA_VISIBLE_DEVICES'] = s
            
            self.cuda_id = self.args.local_rank
            torch.cuda.set_device(self.cuda_id)
                    
        
        assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."

        
