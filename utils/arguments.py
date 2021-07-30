import argparse
import yaml
import sys

def parse():
    parser = argparse.ArgumentParser(
                        description='Comprehensive image classification')
    # ===== Architecture ===== #
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    # ===== Dataset ===== #   
    parser.add_argument('--dataname', default='imagenet', type=str,
                            help='mnist | cifar10 | cifar100| fashionmnist | imagenet')                
    parser.add_argument('--datapath', default = '/datasets/imagenet/', 
                        metavar='DIR', help='path to dataset')
    parser.add_argument('-w', '--workers', default=20, type=int, metavar='N',
                        help='number of data loading workers (default: 20)'
                        'these are different from the processes that '
                        'run the programe. They are just for data loading')     
    parser.add_argument('-b', '--batch-size', default=200, type=int,
                        metavar='N',
                        help='mini-batch size per GPU (default: 224)'
                                'has to be a multiple of 8 to make use of Tensor'
                                'Cores. for a GPU < 16 GB, max batch size is 224')
    # ===== Initialization ===== #
    parser.add_argument('--init-policy', default='kaimingn',
                        type=str, help='kaimingn | kaimingu | xaviern | xavieru')
    parser.add_argument('--init-kaiming-mode', default='fan_in', type=str, help='fan_in | fan_out')
    parser.add_argument('--init-kaiming-nonlinearity', default='relu', type=str, help='nonlinear function')
    parser.add_argument('--init-bias', default='zero', type=str, help='biases initializiation')
    # ===== Optimization ======== #
    parser.add_argument('--optimizer', '-o', default='SGD+M', type=str, help='SGD | SGD+M | Adam')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--initial-epoch', default=0, type=int, metavar='N',
                        help='initial epoch')
    parser.add_argument('--label-smoothing', type=float,
                        help='Label smoothing not to use it, default=0', default=0.1)
    # ===== Learning Rate ======== #                    
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--lr', '--learning-rate', default=0.256, type=float,
                        metavar='LR',
                        help='Initial learning rate.')
    parser.add_argument('--lr-policy', default='constant_lr',
                        help='constant_lr | cosine_lr | normal_lr: policy for the learning rate.')
    parser.add_argument('--warmup-length', default=5, type=int, help='Number of warmup iterations')                            
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='Multistep multiplier')
    parser.add_argument('--lr-steps', default=None, type=int, nargs='+', 
                        help='list of epochs for droping learning rate')
    parser.add_argument('--lowest-lr', default=0.0001, type=float, help='Lowest learning rate for training')
    parser.add_argument('--scale-coslr', default=1, type=float, help='Scale for cosine learning rate')
    parser.add_argument('--exp-coslr', default=1, type=int, 
                        help='Exponential of cosine learning rate (odd interger)')
    parser.add_argument('--normal-exp-scale', default=1500, type=float,
                        help='Scale for normal function for learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--nesterov', help='Nesterov momentum', action='store_true')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    # ===== Logging ======== #
    parser.add_argument('--print-freq-tr', default=10, type=int,
                        metavar='N', help='train print frequency (default: 10)')
    parser.add_argument('--print-freq-ts', default=10, type=int,
                        metavar='N', help='train print frequency (default: 10)')
    parser.add_argument('--config', default='fc_tain', type=str,
                            help='name of yaml configuration file') 
    parser.add_argument('--logterminal', help='logs reports in terminal', action='store_true') 
    parser.add_argument('--save-stages', help='save each stage of the run', action='store_true') 
    # ===== Distributed processing ======== #                   
    parser.add_argument('--gpu-idx', default=None, type=int, nargs='+', 
                        help='indices of GPUs on which a process is run')
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--channels-last', type=bool, default=False)



    args = parser.parse_args()

    with open(f'./configs/{args.dataname}/{args.arch}/{args.config}.yaml', 'r') as f:
        yaml_dict = yaml.load(f, Loader=yaml.FullLoader)
    
    for arg in vars(args):
        if '--'+ arg.replace('_','-') in sys.argv:
            yaml_dict[arg] = getattr(args, arg)

    args.__dict__.update(yaml_dict)

    return args




