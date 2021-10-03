# Datasets
In this document we will explain which datasets are uses and what kind of preprocessing is done before utilizing them.

## Datasets that are used in this repository

- [MNIST](http://yann.lecun.com/exdb/mnist/) 
- [FashionMNIST](https://github.com/zalandoresearch/fashion-mnist)
- [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
- [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html)
- [ImageNet-1K](https://www.image-net.org/)

## What preprocessing is done before using datasets?

- [MNIST](http://yann.lecun.com/exdb/mnist/) 
  - This dataset has the following properties and in order to it normalization is applied using the scaled mean and standard deviation:
    - Min pixel value: 0 
    - Max pixel value: 255
    - Mean pixel value 33.31 
    - Pixel values standard deviation: 78.57
    - Scaled mean: 0.1307 (33.31/255)
    - Scaled standard deviation: 0.3081 (78.57/255)
  - Normalization is also applied for the test

- [FashionMNIST](https://github.com/zalandoresearch/fashion-mnist)
  - This dataset is normalized with mean 0.5 and standard deviation 0.
  - Normalization is also applied for the test

- [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
  - This dataset is normalized by mean and standard deviation of each axis. The mean of axes are (0.4914, 0.4822, 0.4465) and the standard deviation of each axis are given by (0.2023, 0.1994, 0.2010) repectively.
  - Cropping randomly using [`transforms.RandomCrop`](https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.RandomCrop)
  - Flippig Horizontally using [`transforms.RandomHorizontalFlip`](https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.RandomHorizontalFlip)
  - Only normalization is applied for the test

  - [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html)
  - Pixels in each axis are normalized by means of axes as (125.3, 123.0, 113.9) and their standard deviation (63.0, 62.1, 66.7). Every number in these tuples is divied by 255 and then passed in to `transforms.Normalize` function.

    ```{python}
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    transforms.Normalize(mean, std)
    ```
  - Cropping randomly using [`transforms.RandomCrop`](https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.RandomCrop)
  - Flippig Horizontally using [`transforms.RandomHorizontalFlip`](https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.RandomHorizontalFlip)
  - Flipping vertically using [`transforms.RandomVerticalFlip`](https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.RandomVerticalFlip)
  - Changing the brightness, contrast, saturation and hue of an image using [ColorJitter](https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.ColorJitter)
  - [Random affine](https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.RandomAffine)
  - Convert to grayscale randomly using [`transforms.RandomGrayscale`](https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.RandomGrayscale)
  - Only normalization is applied for the test


  - [ImageNet-1K](https://www.image-net.org/)
  - Train

    - Flippig Horizontally using [`transforms.RandomHorizontalFlip`](https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.RandomHorizontalFlip)
    - Crop a random portion of image and resize it to a given size using [`transforms.RandomResizedCrop`](https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.RandomResizedCrop)
  - Test
    - First we resize the input image to the given size using [`transforms.Resize`](https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.Resize)
    - Then, crop the image at the center using [`transforms.CenterCrop`](https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.CenterCrop)

    ### How data is loaded to the model?

*Distributed training:*
Loading data can be easily done by creating an instance of [DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) but since we are using distributed training, we use [DistributedSampler](https://pytorch.org/docs/stable/_modules/torch/utils/data/distributed.html#DistributedSampler). 

*Sampler:*
Sampler get a subseto of data and use it for training. Because of that we use [train_sampler.set_epoch(epoch)](https://github.com/pytorch/examples/issues/501#issuecomment-458808573) while training to make suere same samples are fetched to workers across multiple runs even across distributed training. The same explanation is [here](https://discuss.pytorch.org/t/how-dose-distributed-sampler-passes-the-value-epoch-to-data-loader/110929).

*collate_fn*:

Also, a [`collate_fn`](https://stackoverflow.com/a/65875359/11748994) is also used when we need to do some preprocessing on the data to create a batch. For example [here](https://discuss.pytorch.org/t/questions-about-dataloader-and-dataset/806/2) is suggested to use `collate_fn` function to handle situations that cannot be handle by `DataLoader`. Or, if you want to create a [dataloader with variable-size input](https://discuss.pytorch.org/t/how-to-create-a-dataloader-with-variable-size-input/8278). In addition, [this post](https://discuss.pytorch.org/t/how-to-use-collate-fn/27181/4) breifly explains it but detailed version is [here](https://discuss.pytorch.org/t/how-to-create-batches-of-a-list-of-varying-dimension-tensors/50773/14). For official documentation, please refer [here](https://pytorch.org/docs/stable/data.html#dataloader-collate-fn).

*DataPrefetcher:*

After above preprocessing we fetch the data using `DataPrefetcher` class.

## What modules are used to prepare the data?

### `main.py`
### `utils/datasets.py`
### `utils/train_validation.py`















# Datasets
In this document we will explain which datasets are uses and what kind of preprocessing is done before utilizing them.

## Datasets that are used in this repository

- MNIST

Complete them from the paper and cite them.
