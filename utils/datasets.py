import os, torch
import numpy as np
from torchvision import datasets, transforms, utils

def get_data(args):

    dataname, datapath = args.dataname, args.datapath

    if dataname == 'mnist':
        
        transform = transforms.Compose([transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(datapath, 
                                    train=True, 
                                    download=True, 
                                    transform=transform)

        validation_dataset = datasets.MNIST(datapath, 
                                            train=False, 
                                            transform=transform)
    elif dataname == 'cifar10':
    
        transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

        transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),]) 

        train_dataset = datasets.CIFAR10(datapath, 
                                    train=True, 
                                    download=True, 
                                    transform=transform_train)

        validation_dataset = datasets.CIFAR10(datapath, 
                                            train=False, 
                                            transform=transform_test)

    elif dataname == 'cifar100':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]

        train_transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(),
             transforms.RandomCrop(32, padding=4),
             transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.4, hue=0.4),
             transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.5, 1.25),
                                     shear=5, resample=False, fillcolor=0),
             transforms.RandomGrayscale(),
             transforms.RandomVerticalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean, std)])

        test_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean, std)])
        
        train_dataset = datasets.CIFAR100(
                                    root=datapath,
                                    train=True,
                                    download=True,
                                    transform=train_transform )
                                    
        validation_dataset = datasets.CIFAR100(
                                    root=datapath,
                                    train=False,
                                    download=True,
                                    transform=test_transform)
                                    
    elif dataname == 'fashionmnist':

        transform = transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize((0.5,), (0.5,))])

        train_dataset = datasets.FashionMNIST(datapath, 
                                    train=True, 
                                    download=True, 
                                    transform=transform)
                                

        validation_dataset = datasets.FashionMNIST(datapath, 
                                            train=False, 
                                            transform=transform)

    elif dataname == 'imagenet':

        # Data loading code
        traindir = os.path.join(args.datapath, 'train')
        valdir = os.path.join(args.datapath, 'val')

        if args.arch == "inception_v3":
            raise RuntimeError("Currently, inception_v3 is not supported by this example.")
        else:
            crop_size = 224
            val_size = 256

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(crop_size),
                transforms.RandomHorizontalFlip(),
                # transforms.ToTensor(), Too slow
                # normalize,
            ]))
        validation_dataset = datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(val_size),
                transforms.CenterCrop(crop_size),
            ]))


    return train_dataset, validation_dataset
    
def loaders(train_dataset, validation_dataset, args):
    # makes sure that each process gets a different slice of the training data
    # during distributed training
    train_sampler = None
    val_sampler = None
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(validation_dataset)

    if args.channels_last:
        memory_format = torch.channels_last
    else:
        memory_format = torch.contiguous_format

    collate_fn = lambda b: fast_collate(b, memory_format)

    # notice we turn off shuffling and use distributed data sampler
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, 
        shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True,
        sampler=train_sampler, 
        collate_fn=collate_fn if args.dataname == 'imagenet' else None
        )

    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        sampler=val_sampler,
        collate_fn=collate_fn if args.dataname == 'imagenet' else None
        )

    return train_loader, validation_loader, train_sampler


def fast_collate(batch, memory_format):

    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    tensor = torch.zeros( (len(imgs), 3, h, w), dtype=torch.uint8).contiguous(memory_format=memory_format)
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        if nump_array.ndim < 3:
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)
        array_copy = nump_array.copy()
        tensor[i] += torch.from_numpy(array_copy)
    return tensor, targets



        
