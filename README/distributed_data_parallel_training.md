# Pytorch Distributed Data Parallel training
 
To run multiple processes on different machines and various GPUs, our code uses the [Pytorch Distributed Data Parallel](https://pytorch.org/docs/stable/notes/ddp.html) class which is a Pytorch class. In this document we will go through what are the advantages of using Pytorch Distributed Data Parallel training and how one can use it.
 
**Remark1:**
Note that [Pytorch `DataParallel` class](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html) is [not as capable as](https://discuss.pytorch.org/t/dataparallel-vs-distributeddataparallel/77891/4) [`DistributedDataParallel`](https://pytorch.org/docs/stable/notes/ddp.html) and [Pytorch `DATAPARALLEL` documentation](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html) recommends it `DistributedDataParallel`. 
 
## Why do we use the Pytorch Distributed Data Parallel class?
- `DataParallel` cannot scale beyond one machine. It is slower than `DistributedDataParallel` even in a single machine with multiple GPUs due to Global Interpreter Lock (GIL) contention across multiple threads and the extra overhead introduced by scatter and gather and per-iteration model replication.
- Not only the number of GPUs can be specified but also IDs of GPUs can be passed to the code to use unallocated GPUs. For example the following flag tells the code that only use GPUs with the specified IDs:
 
```{python}
--gpu-ids 1,6,3
```
 
## How to use Pytorch Distributed Data Parallel class?
 
### Shell commands for doing Pytorch Distributed Data Parallel training
 
In order to apply the distributed data parallel processing, one needs to execute the following lines in terminal while executing [`main.py`](https://github.com/sdamadi/image-classification/blob/main/main.py) module:
 
```{shell}
$ python -m torch.distributed.launch --nproc_per_node=1 \
$ --master_port=$RANDOM main.py \
$ --gpu-idx 1,6,3\
```
 
We will go through each flag as the following:
 
- To apply the distributed data parallel training one needs to use [`-m` command-line flag](https://stackoverflow.com/a/22250157/11748994) to import `torch.distributed.launch`. When you use the `-m` command-line flag, Python will import a module or package for you, then run it as a script. When you don't use the `-m` flag, the file you named is run as just a script. In this case we are importing `torch.distributed.launch` at the same time that we are executing the [`main.py`](https://github.com/sdamadi/image-classification/blob/main/main.py) file.
 
- Pass the number of processes that you wish to run by passing `--nproc_per_node=3` in. In the above case we want our neural network to be run on 3 GPUs `1,6,3` simultaneously.
 
- Passing `--master_port=$RANDOM` is not necessary but without it you may not be able to get an open master port. However, passing `--master_port=$RANDOM` makes choosing the master port random so having an open master port is guaranteed.
 
- Specify those 3 processes that want to run your network on by passing '--gpu-idx 1,6,3'.
 
- `\` breaks line in command line so you can see what you are executing.
 
### Which modules are the ones containing classes and code snippets that implement Pytorch Distributed Data Parallel training?
 
For using Pytorch Distributed Data Parallel class we use the following modules:
 
- [`main.py`](https://github.com/sdamadi/image-classification/blob/main/main.py)
- [`utils/distributed.py`](https://github.com/sdamadi/image-classification/blob/main/utils/distributed.py).
 
 
### [`main.py`](https://github.com/sdamadi/image-classification/blob/main/main.py)
 
[`main.py`](https://github.com/sdamadi/image-classification/blob/main/main.py) creates an instance of `Distributed` class which is in `utils/distributed.py`. To create that instance we use the following:
 
```{python}
distributed = Distributed(args)
```
Once you create `distributed` instance will have the following attributes for `distributed` instance:
 
- Size of the world which is the number of GPUs that are being utilized during your training, i.e., `distributed.args.world_size`
- Status of whether we are using the distributed training, i.e., `distributed.args.distributed`. When we set `--nproc_per_node=1` this will be `False` since the training process is done on one single GPU. Otherwise it is set to `True`.
- The list of GPUs that are being utilized is another attribute of `distributed` incestance, i.e., `distributed.args.distributed`
- Finally, we keep the id of a GPU that is being used to run the current process, i.e., `distributed.cuda_id`
 
Once we have all these attributes, we use the following assignments:
 
```{Python}
args.world_size, args.distributed = distributed.args.world_size, distributed.args.distributed
args.gpu_ids, cuda_id = distributed.args.gpu_ids, distributed.cuda_id
```
 
Every argument is used for a different purpose as follows:
 
- `args.world_size` is used to name [the current scenario](https://github.com/sdamadi/image-classification/blob/main/README/scenario_name.md). 
 
- `args.distributed` is used to decide whether we want to use `` or not as the following code snippet shows:
 
```{python}
if args.distributed:
  model = DDP(net, device_ids = [cuda_id], output_device = cuda_id)
```
 
- `cuda_id` or `args.local_rank` determines whether [directories](), [logger files](), and [tensorboard events]() should be created. All of the aforementioned happens when `args.local_rank` is equal to zero. That means only for the zeroth process, we create necessary directories, logger files, and tensorboard evensts. To see how these changes are made look at [`utils/writer.py`](https://github.com/sdamadi/image-classification/blob/main/README/writer.md)
 
- `args.gpu_ids` is used to print out which GPUs are allocated for training the current run.
 
## How [`utils/distributed.py`](https://github.com/sdamadi/image-classification/blob/main/utils/distributed.py) module works?
 
This module as we explained in the above creates a `Distributed` object. This class is initialized as follows:
 
-  `self.args.distributed` is initialized to be `False`. This is because in a case where `--nproc_per_node=1`, one wants the training to be run on a single GPU. Therefore, the code executes the else part of `if self.args.distributed`.
- In a case where `--nproc_per_node>1`, `self.args.distributed` is initialized to `True`. This is checked as follows:
 
```{python}
if 'WORLD_SIZE' in os.environ:
  self.args.distributed = int(os.environ['WORLD_SIZE']) > 1
```
 
Note that in the second case the if part of `if self.args.distributed` is executed and the following would be executed:
 
- [`torch.distributed.init_process_group` function](https://pytorch.org/docs/stable/_modules/torch/distributed/distributed_c10d.html#init_process_group) is called. As a result the default distributed process group is initialized and this will also initialize the distributed package. 
- Each process is ranked automatically as [`torch.distributed.init_process_group` function](https://pytorch.org/docs/stable/_modules/torch/distributed/distributed_c10d.html#init_process_group) determines.
 
The following summarizes the above:
 
```{pyton}
torch.distributed.init_process_group(backend='nccl', init_method='env://')
self.args.world_size = torch.distributed.get_world_size()
self.args.local_rank = torch.distributed.get_rank()
self.cuda_id = self.args.local_rank 
```
Finally, we set the GPU id, i.e., `torch.cuda.set_device(self.cuda_id)`.