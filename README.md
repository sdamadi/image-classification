# Comprehensive distributed image classification using Pytorch
By using this repository you can train traditional as well as the state of the art models (networks) on different image datasets to train an image classifier.

## What are the advantages of this repository?
1. All trainings are done using [native Pytorch Automatic Mixed Precision (AMP)](https://pytorch.org/blog/accelerating-training-on-nvidia-gpus-with-pytorch-automatic-mixed-precision/) which is much faster.
2. Training can be done distributed on different GPUs or be done on a single GPU.
3. Not only the number of GPUs can be specified but also GPU ids can be passed to the code to use unallocated GPUs
4. The history of the training and test is logged and saved in a folder like `./history/logs/cifar10/resnet34/time`.
  - `Time` is the time when code is run and is in `y-m-d-hh-mm-ss`.
  - The corresponding [TENSORBOARD file](https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html) is save as `./runs/cifar10/resnet34/time` to keep track of experiments.
5. Learning rate can be chosen to be `Cosine`, `Multisteps`, or `Constant` with warmup feature.
6. Each scenario (experiment) has a yaml file located in `configs` folder where hyper parameters can be set. For example to train a fully connected network on MNIST you can pass the following:
```
python -m torch.distributed.launch --nproc_per_node=1 --master_port=$RANDOM main.py -a fc --dataname mnist --config mnist_fc_train --logterminal
```
If you do not pass any argument except for the above ones, other hyper parameters are read from yaml file `./configs/mnist/fc/mnist_fc_train.yaml`

