## Comprehensive image classification library 
### Why you should this repository?

- [Native Pytorch Automatic Mixed Precision (AMP)](https://github.com/sdamadi/image-classification/blob/main/README/automatic_mixed_precision.md) 
- Decentralization of training on multiple GPUs
- Assigning folders named with the time of execution for the followings:
  - Training and test logs as text
  - Initial and final checkpoints
  - Training and test statistices as Numpy variables
  - Tensorboard logs
- Various learning rate schedulers
- Hyperparamters are given as a `config` file


This repository is meant to serve people who want to generate results of image classification and [compare different outputs](put a table and explain it is the average of five runs) on [various networks and datasets](https://github.com/sdamadi/image-classification/blob/main/README/combinations.md). The following are capabilities of this repository:

1. It utilizes [native Pytorch Automatic Mixed Precision (AMP)](https://pytorch.org/blog/accelerating-training-on-nvidia-gpus-with-pytorch-automatic-mixed-precision/) for fast training (the speed is doubled at least). 
2. To run multiple processes on different machines and various GPUs, it uses [Pytorch Distributed Data Parallel](https://pytorch.org/docs/stable/notes/ddp.html) class. Note that [Pytorch DataParallel](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html) is [not as capable as](https://discuss.pytorch.org/t/dataparallel-vs-distributeddataparallel/77891/4) [Distributed Data Parallel](https://pytorch.org/docs/stable/notes/ddp.html) and [Pytorch Distributed Data Parallel document(https://pytorch.org/docs/stable/notes/ddp.html) recommends it. Not only the number of GPUs can be specified but also IDs of GPUs can be passed to the code to use unallocated GPUs.
3. It creates a [*history*](https://github.com/sdamadi/image-classification/blob/main/README/historytree.md) folder including [*logs of training*](logs), [*initialization*](initialization) *and* [*trained*](trained) *networks*, and [*statistics of training*](variables).
4. All saved files are named such that the time of execution: [`yyyy_MM_dd_hh_mm_ss`](tree explain and folders) is at the begining of the file.
7. [Curves of statistics of training and evaluation](link) which is average of five scenarios [can be found here.](Google Colabe to generate them).
8. The code saves a [TENSORBOARD file](https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html) with [the same name of log file](tree) to avoid confusions.
9. There are [four learning rate strategies](Colab to impliment) namely *constant*, *cosine*, *multisteps*, and *noraml* (decaying like normal distribution) all equipted with warmup feature.
10. For [each scenario](page including networks as above and link to Google Drive) there are five trained network with their initialization.
11. Hyperparameters for every scenarion has a [`yaml` configuratin](tree) file which can be set before training. Also, [Hyperparameters](table) are provided in one place for whom want to replicate the results. 
12. There is an option to run differnet scenarios without changing confuguration files using runners.
13. are also runner files where you can use them and change some variables and then run them to get your code run. There are runner files that can run multiple processes in the background and we can just average their outputs out to report loss and accuracy. They are located in ….


#### The primitive case

An example for training a fully connected network on MNIST dataset is the following:

```{bash}
python -m torch.distributed.launch --nproc_per_node=1 --master_port=$RANDOM main.py \

-a fc --dataname mnist --config mnist_fc_train --logterminal
```

#### The advacnced case

An example for training a ResNet50 network on ImageNet1K dataset is the following:

```{bash}
python -m torch.distributed.launch --nproc_per_node=5 --master_port=$RANDOM main.py 
-a resnet50 --dataname imagenet --config imagenet_resnet50_train

```

When the code is run for training a network on a dataset the following `history` folder is created

To fulfill this purpose, this repository has several feature like automatic dataset download, automatic training log, 







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

