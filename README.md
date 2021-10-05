## Comprehensive image classification library
 
### Why image classification?
 
Image classification is one of the fundamental computer vision tasks that serves as the backbone for solving different computer vision problems.
 
### How does this repository help?
 
- [Native Pytorch Automatic Mixed Precision (AMP)](https://github.com/sdamadi/image-classification/blob/main/README/automatic_mixed_precision.md) 
- [Distributed data_parallel training (Pytorch Distributed Data Parallel)](https://github.com/sdamadi/image-classification/blob/main/README/distributed_data_parallel_training.md)
  - Run multiple processes on different machines and various GPUs

- [Three different initialization](https://github.com/sdamadi/image-classification/blob/main/README/initialization.md) are supported.
- [Hyperparamters](https://github.com/sdamadi/image-classification/blob/main/README/combinations.md) are given as a [`Yaml`]() file for each scenario.
 
- Creating folders [named](https://github.com/sdamadi/image-classification/blob/main/README/scenario_name.md) with the time of execution for the followings outputs:
  - Training and test logs as text
  - Initial and final checkpoints
  - Training and test statistics as Numpy variables
  - Tensorboard logs
- [Four different learning rate schedulers](https://github.com/sdamadi/image-classification/blob/main/README/learning_rate.md) are supported.
- [Hyperparamters](https://github.com/sdamadi/image-classification/blob/main/README/combinations.md) are given as a [`Yaml`]() file for each scenario.
- All training and validation [metrics](https://github.com/sdamadi/image-classification/blob/main/README/training_test_metrics.md) are save for further analysis.

- [Tensorboard](https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html) events are logged in a separate folder categorized by the dataset and architecture.
- Every run file is given as bash file. 
 
#### The primitive case
 
An example for training a fully connected network on MNIST dataset is the following:
 
```{bash}
python -m torch.distributed.launch --nproc_per_node=1 --master_port=$RANDOM main.py \
-a fc --dataname mnist --config mnist_fc_train --logterminal
```
 
#### The advanced case
 
An example for training a ResNet50 network on ImageNet1K dataset is the following:
 
```{bash}
python -m torch.distributed.launch --nproc_per_node=5 --master_port=$RANDOM main.py 
-a resnet50 --dataname imagenet --config imagenet_resnet50_train
 
```