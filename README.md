# Small size datasets for running ASNI
- MNIST, FashionMNIST, CIFAR-10, CIFAR-100

# Datasets
The following datasets can be run using this code.


# Different scenarios
Each scenario has its own `Yaml` file that should be passed when a specific scenario is going to run. For example, one can use the following to train an `FC` network on `MNIST` dataset:

`python -m torch.distributed.launch --nproc_per_node=1 --master_port=$RANDOM main.py -a fc --dataname mnist --config mnist_fc_train`

The general `Yaml` files are as follows for each combination of networks and datasets:

| policy    |      Description      | 
|:----------|:----------------------|
| train     |                       |
| asni      |                       |
| prepruned |                       |
| quantized |                       |
| lottery   |                       |
| str       |                       |

