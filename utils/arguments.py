import argparse
import yaml
import sys
 
def parse():
    parser = argparse.ArgumentParser(
                        description='Comprehensive image classification')
    # ===== Default configuration ===== #
    parser.add_argument('--config', default='mnist_fc_train', type=str,
                        help='name of yaml configuration file') 
    # ===== Architecture ===== #
    parser.add_argument('--arch', '-a', #dest='arch',
                        metavar='ARCH', default='resnet50')
    # ===== Dataset ===== #   
    parser.add_argument('--dataname', #dest='data_name',
                        default='imagenet', type=str,
                        help='mnist | cifar10 | cifar100| fashionmnist | imagenet')                
    parser.add_argument('--datapath', #dest='data_path',
                        default = '/datasets/imagenet/', 
                        metavar='DIR', help='path to dataset')
    parser.add_argument('-w', '--workers', #dest='data_workers',
                        default=20, type=int, metavar='N',
                        help='number of data loading workers (default: 20)'
                        'these are different from the processes that '
                        'run the program. They are just for data loading')     
    parser.add_argument('-b', '--batch-size', #dest='data_bsize',
                        default=200, type=int,
                        metavar='N',
                        help='mini-batch size per GPU (default: 224)'
                                'has to be a multiple of 8 to make use of Tensor'
                                'Cores. for a GPU < 16 GB, max batch size is 224')
    parser.add_argument('--channels-last', #dest='data_channel',
                        type=bool, default=False)
    # ===== Model Initialization ===== #
    parser.add_argument('--dense-init-policy', #dest='init_dense_policy',
                        default='kaimingn',
                        type=str, help='default | kaimingn | kaimingu | xaviern | xavieru')
    parser.add_argument('--dense-bias-init', #dest='init_bias_policy',
                        default='zero', type=str, 
                        help='dense bias initialization: zero | kaimingn | kaimingu | xaviern | xavieru')
    parser.add_argument('--kaiming-mode', #dest='init_kaiming_mode',
                        default='fan_in', type=str, help='fan_in | fan_out')
    parser.add_argument('--kaiming-nonlinearity', #dest='init_kaiming_nonlinearity',
                        default='relu', type=str, help='nonlinear function')
    # ===== Optimization ======== #
    parser.add_argument('--optimizer', '-o', #dest='optimizer_alg',
                        default='SGD+M', type=str, help='SGD | SGD+M | Adam')
    parser.add_argument('--initial-epoch', #dest='optimizer_init_epoch',
                        type=int, default=0,
                        help='initial epoch of training')
    parser.add_argument('--epochs', #dest='optimizer_epochs',
                        default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--momentum', #dest='optimizer_momentum',
                        default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--nesterov', #dest='optimizer_nesterov',
                        help='Nesterov momentum', action='store_true')
    parser.add_argument('--label-smoothing', #dest='optimizer_epochs',
                        type=float, default=0,
                        help='Label smoothing not to use it, default=0')
    # ===== Learning rate ======== #
    parser.add_argument('--lr', '--learning-rate', #dest='lr_init_value',
                        default=0.256, type=float, metavar='LR', 
                        help='Initial learning rate.')
    parser.add_argument('--lr-policy', #dest='lr_policy',
                        default='constant_lr',
                        help='constant_lr | cosine_lr | normal_lr: policy for the learning rate.')
    parser.add_argument('--weight-decay', '--wd', #dest='lr_weight_decay',
                        default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--warmup-length', #dest='lr_warmup_length',
                        default=5, type=int, help='Number of warmup iterations')
    # ===== Multistep damped learning rate ======== #                            
    parser.add_argument('--lr-gamma', #dest='lr_damp_rate',
                        default=0.1, type=float, help='Damp rate for learning rate')
    parser.add_argument('--lr-steps', dest='lr_damp_steps',
                        default=None, type=int, nargs='+', 
                        help='list of epochs for damping learning rate')
    # ===== Cosine damped learning rate ======== # 
    parser.add_argument('--scale-coslr', #dest='lr_cosine_scale',
                        default=1, type=float, help='Scale for cosine learning rate')                    
    # ===== Gaussian damped learning rate ======== # 
    parser.add_argument('--scale-gausslr', #dest='lr_gauss_scale',
                        default=1500, type=float,
                        help='Scale for normal function for learning rate')    
    # ===== Logging ======== #
    parser.add_argument('--print-freq-tr', #dest='log_print_train',
                        default=10, type=int, metavar='N',
                        help='train print frequency (default: 10)')
    parser.add_argument('--print-freq-ts', #dest='log_print_test',
                        default=10, type=int, metavar='N',
                        help='train print frequency (default: 10)')
    parser.add_argument('--logterminal', #dest='log_terminal',
                        help='logs reports in terminal', action='store_true') 
    parser.add_argument('--save-epochs', #dest='log_save_epochs',
                        help='save the model after each epoch of training', action='store_true') 
    # ===== Training Status ===== #
    parser.add_argument('--pretrained', #dest='pretrained', 
                        action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--resume', #dest='resume', 
                        default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--evaluate', '-e',  #dest='evaluate',
                        action='store_true',
                        help='evaluate model on validation set')
    # ===== Distributed processing ======== #                   
    parser.add_argument('--gpu-ids', #dest='dist_gpus',
                        default=None, type=int, nargs='+', 
                        help='indices of GPUs on which a process is run')
    parser.add_argument('--local_rank', #dest='dist_local_rank',
                        default=0, type=int)
 
    args = parser.parse_args()
 
    with open(f'./configs/{args.dataname}/{args.arch}/{args.config}.yaml', 'r') as f:
        yaml_dict = yaml.load(f, Loader=yaml.FullLoader)
    
    for arg in vars(args):
        if '--'+ arg.replace('_','-') in sys.argv:
            yaml_dict[arg] = getattr(args, arg)
 
    args.__dict__.update(yaml_dict)
 
    return args
 

