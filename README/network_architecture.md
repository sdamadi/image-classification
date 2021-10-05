# Network 
 
In this document we will explain which network architectures are used to train an image classifier.
 
## Network architectures
 
- Fully-connected network
- Small convolutional neural networks
- Residual neural network
- MobileNet
- VGG neural networks
 
### Fully-connected network
Fully connected networks are not used so often. One one the famous fully connected neural networks is [LeNet-300-100](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf). This network is a 3-layer network that can learn [MNIST](http://yann.lecun.com/exdb/mnist/) dataset.
 
### Small convolutional neural networks
Shallow convolutional neural networks with only 2/4/6 convolutional layers can also be used to learn [MNIST](http://yann.lecun.com/exdb/mnist/)  and [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) datasets. These networks are introduced in [The Lottery Ticket Hypothesis](https://arxiv.org/abs/1803.03635) paper to investigate the pruning algorithm introduced in the paper.
### Residual neural network
The most famous and often used neural networks are residual neural networks introduced [here](https://arxiv.org/abs/1512.03385). We use ResNet-18/35/50 for learning [ImageNet-1K](https://www.image-net.org/).
### MobileNet
MobileNet is a compressed neural network optimized for mobile devices. This model delivers high accuracy while keeping the parameters and mathematical operations as low as possible. We use [MobileNetV2](https://openaccess.thecvf.com/content_cvpr_2018/papers/Sandler_MobileNetV2_Inverted_Residuals_CVPR_2018_paper.pdf).
### VGG neural networks
We utilize [VGG](https://arxiv.org/abs/1409.1556)-style networks named as 284 VGG-11/13/16/19 with batch normalization and average pooling after convolution layers followed by a fully connected layer. This decreases the parameter count of original VGG networks and mitigates
the parameter inefficiency with VGG networks.
 
 
 
## What modules are used for creating the network?
 
### [`main.py`](https://github.com/sdamadi/image-classification/blob/main/main.py)
In this module we create an instance of [`Archs`](https://github.com/sdamadi/image-classification/blob/main/utils/archs.py) class as follows:
```{python}
net = Archs(args).model.cuda(cuda_id)
``` 
The argument including the model architecture determines which network architecture should be used. Once we have the network as our model, it should be properly initialized. This step is explained in [initialization document](https://github.com/sdamadi/image-classification/blob/main/README/initialization.md).
 
### [`utils/archs.py`](https://github.com/sdamadi/image-classification/blob/main/utils/archs.py).
 
### [`models/bare`](https://github.com/sdamadi/image-classification/tree/main/models/bare)
This module includes every network architecture separately.
