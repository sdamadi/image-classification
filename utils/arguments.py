import argparse
import yaml
import sys

def parse():
    parser = argparse.ArgumentParser(
                        description='Amenable Sparse Network Investigator')
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
    parser.add_argument('--label-smoothing', type=float,
                        help='Label smoothing not to use it, default=0', default=0.1)
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
    # ===== Pruning ======== #
    parser.add_argument('--initial-stage', type=int, default=0,
                        help='initial stage of pruning (useful on restarts)')
    parser.add_argument('--stages', '-s',type=int, help='total stages for pruning or training', default=2)
    parser.add_argument('--prepruned-model', action='store_true',
                        help='whether pruned model should be traind or not') 
    parser.add_argument('--quantize-prepruned', action='store_true',
                        help='whether the prepruned model should be quantized to c+ and c_')
    parser.add_argument('--quantize-bias', action='store_true',
                        help='whether biases of the prepruned model should be quantized or not')  
    parser.add_argument('--prepruned-scen', default='', type=str,
                            help='the date and time of prepruned scenario')
    parser.add_argument('--pruning-strategy', default='asni', type=str,
                            help='prepruned | quantize | asni | lottery | str | snip')
    parser.add_argument('--nonpruned-percent', default=99.8, type=float, metavar='M',
                        help='Percentage of nonpruned parameters')
    parser.add_argument('--mask-stage', default=0, type=int, metavar='M',
                        help='stage whose mask is used for reinitialization')
    parser.add_argument('--prune-bn', help='prune batch norm elements', action='store_true')
    parser.add_argument('--prune-bias', help='prune bias elements', action='store_true')
    parser.add_argument('--local-quant', help='quantization type (network-wise|layer-wise)',
    action='store_true')
    # ===== Asni ======== #
    parser.add_argument('--asni-perc-max', default=100, type=float,
                        help='maximum allowable pruning percentage')
    parser.add_argument('--asni-mode', default='sigmoid', type=str,
                            help='sine | sigmoid')
    parser.add_argument('--asni-rest-stage', default=1000, type=float,
                        help='pruning is decreased sharply')                         
    parser.add_argument('--asni-sin-scale', default=1, type=float, metavar='M', 
                        help='scaling sin function for pruning percentage')
    parser.add_argument('--asni-sin-exponent', default=1, type=int, metavar='M', 
                        help='Exponent of sin function for pruning percentage')
    parser.add_argument('--asni-sin-mag', default=100, type=float,
                        help='Magnitude of sine function')   
    parser.add_argument('--asni-sigmoid-scale-1', default=1, type=float, metavar='M', 
                        help='scaling Sigmoid function for pruning percentage')
    parser.add_argument('--asni-sigmoid-trans-1', default=0.5, type=float, metavar='M', 
                        help='fraction of stages to translate Sigmoid function')
    parser.add_argument('--asni-sigmoid-mag-1', default=100, type=float,
                        help='Magnitude of Sigmoid function') 
    parser.add_argument('--asni-sigmoid-scale-2', default=1, type=float, metavar='M', 
                        help='scaling Sigmoid function for pruning percentage')
    parser.add_argument('--asni-sigmoid-trans-2', default=0.5, type=float, metavar='M', 
                        help='fraction of stages to translate Sigmoid function')
    parser.add_argument('--asni-sigmoid-mag-2', default=100, type=float,
                        help='Magnitude of Sigmoid function') 
                                          
    # ===== Lottery ======== #                        
    parser.add_argument('--percent', type=int, default=10,
                        help='pruning percentile rank')
    parser.add_argument('--local-prune', help='pruning type (network-wise|layer-wise)',
                            action='store_true')
    # ===== STR ======== #
    parser.add_argument('--init-threshold', default=-5, type=float, metavar='M',
                        help='STR initial threshold') 
    parser.add_argument('--init-threshold-type', default='constant', type=str,
                            help='STR initialization for threshold')
    parser.add_argument('--str-nonlinear', default='sigmoid', type=str,
                            help='sigmoid | none')
    parser.add_argument('--str-activation', default='relu', type=str,
                            help='STR activation for thresholding')                         
    # ===== GMP ======== #
    # ===== RigL ======== #
    # ===== SNIP ======== #    
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




