import torch
import torchvision.models as torch_models
from models import *

class Archs(object):
    def __init__(self, args):
        self.args = args
        if args.pruning_strategy in {'asni', 'lottery', 'quantize', 'prepruned'}:
            if args.arch == 'fc':
                model = fc.FC()
            elif 'conv' in args.arch:
                in_ch = 1 if args.dataname in {'mnist', 'fashionmnist'} else 3
                imgsz = 28 if args.dataname in {'mnist', 'fashionmnist'} else 32
                if args.arch == 'conv2a':
                    model = conv2a.Conv2A(in_ch, imgsz)
                elif args.arch == 'conv4a':
                    model = conv4a.Conv4A(in_ch, imgsz)
                elif args.arch == 'conv6a':
                    model = conv6a.Conv6A(in_ch, imgsz)
                
                elif args.arch == 'conv2':
                    model = conv2.Conv2(in_ch, imgsz)  
                elif args.arch == 'conv4':
                    model = conv4.Conv4(in_ch, imgsz) 
                elif args.arch == 'conv6':
                    model = conv6.Conv6(in_ch, imgsz)
            
            elif 'vgg' in args.arch:
                num_classes = 10 if args.dataname in {'cifar10'} else 100
                model = vgg.VGG(args.arch, num_classes = num_classes)

            elif 'resnet' in args.arch or 'mobile' in args.arch:
                model = torch_models.__dict__[args.arch]()

            
            if self.args.local_rank == 0:
                print(f'=> The model, i.e., {args.arch}, '\
                    f'is being replicated on {args.world_size} processes.\n')
        
        elif args.pruning_strategy in 'str':
            if args.arch == 'fc':
                model = strfc.STRFC(str_args = args)
            elif 'vgg' in args.arch:
                num_classes = 10 if args.dataname in {'cifar10'} else 100
                model = strvgg.STRVGG(str_args = args, vgg_name=args.arch, num_classes=num_classes)#, s

            elif 'resnet' in args.arch:
                model = strresnet.__dict__[args.arch](str_args=str_args)


            if self.args.local_rank == 0:
                print(f'=> The model, i.e., {args.arch}, '\
                    f'is being replicated on {args.world_size} processes.\n')
        
        self.model = model.cuda(self.args.local_rank)