# Scenario names

In this document we will go through the advantage of using different names for each run of training (scenario) which. Each scenario has a main name and a specific name. First we will explain why the time of execution is the main name and then we will detail hyper-parameters that constitute the specific name.

## Why do we use the time of execution as the main name of the scenario?

When each run or scenario is executed, the time of execution in seconds is used as the main name of that scenario. This is because it is a unique identifier for each process as long as processes are run at least one second after one another. Note that when only one GPU is used, there is only one time of execution. However, when `--nproc_per_node` is larger than one we are dealing with [Pytorch Distributed Data Parallel training](https://github.com/sdamadi/image-classification/blob/main/README/distributed_data_parallel_training.me). Therefore, there are more than one time of execution. To avoid any confusion, as explained in [Pytorch Distributed Data Parallel training](https://github.com/sdamadi/image-classification/blob/main/README/distributed_data_parallel_training.me) and [How to write outputs](https://github.com/sdamadi/image-classification/blob/main/README/How_to_write_outputs.md), we write outputs when `args.local_rank` is equal to zero which means the execuion time of the zeroth process is always used for a scenarion.

### What is the format of the main name?

As we explained the main name of each scenario is the execution time so this time is formated as `yyyy_MM_dd_hh_mm_ss`. The month, day, hour, minutes, and seconds need not be a two-digit number. For example this `2021_6_3_3_37_16` refers to a scenario that has been run on July 3rd at 3:37:17am.

### Where the main name is used?

There are five places where the main name is used as follow:
- [Logging of the terminal](https://github.com/sdamadi/image-classification/blob/main/README/logging_terminal.md) while the code is being executed
- Saving [the model at initialization](https://github.com/sdamadi/image-classification/blob/main/README/model_at_initialization.md)
- [Saving the model](https://github.com/sdamadi/image-classification/blob/main/README/saving_model.md) after each epoch
- Saving [training and test metrics](https://github.com/sdamadi/image-classification/blob/main/README/training_test_metrics.md) 

- Saving [Tensorboard events](https://github.com/sdamadi/image-classification/blob/main/README/tensorboard.md).

## How the specific name of the scenario is created?

The specific name of each scenario is meant to include the most important [hyperparameters](https://github.com/sdamadi/image-classification/blob/main/README/hyperparameters.md) as follows:

- [Name of the dataset](https://github.com/sdamadi/image-classification/blob/main/README/datasets.md)
- [Architechture of the neural network](https://github.com/sdamadi/image-classification/blob/main/README/network_architecture.md) that is being used
- [Initialization approach](https://github.com/sdamadi/image-classification/blob/main/README/datasets.md)
- Number of GPUs
- Number of mini-batches
- Optimization method
- [Learning rate and learning rate policy](https://github.com/sdamadi/image-classification/blob/main/README/learning_rate.md)
- Number of epochs

As an example for an specific name we have the following:

```
data_cifar10_arch_conv2_init_kaimingn_GPUs_2_minib_60_opt_Adam_lr_0.0002_lrpolicy_constant_lr_ep_20
```
where data is [CIFAR10](will be added), [architecture of the network](will be added) is [CONV2](will be added), [the initialization method](will be added) is [Kaiming](will be added), the number of GPUs is 2, the number of mini-batches is 60, the optimization algorithm is [ADAM](will be added), the learning rate is 0.0002 which will be constant throughout the training because the policy is constant, and the number of epochs is 20.


## Which modules are used to create the main and specific name?

### [`main.py`](https://github.com/sdamadi/image-classification/blob/main/main.py)

In [`main.py`](https://github.com/sdamadi/image-classification/blob/main/main.py) we will create a scenario instance for [`Scenario` class](https://github.com/sdamadi/image-classification/blob/main/utils/scenario.py) that takes care of the main and specific name of the running scenario. This is done as follows:

```
scenario = Scenario(args)
```

Instances of [`Scenario` class](https://github.com/sdamadi/image-classification/blob/main/utils/scenario.py) (e.g. `scenario`) have only two attributes, one is the main name (the time of execution of the first process) which is kept in `scenario.time`; the other one is the main name followed by the specific name which is kept in `scenario.curr_scen_name`. 

## Which modules use instances of [`Scenario` class](https://github.com/sdamadi/image-classification/blob/main/utils/scenario.py)?


- [`utils/writer.py`](https://github.com/sdamadi/image-classification/blob/main/utils/writer.py)

As we explained in [Distributed Data Parallel training](https://github.com/sdamadi/image-classification/blob/main/README/distributed_data_parallel_training.me) when `args.local_rank` is zero we save outputs. Therefore, in [`main.py`](https://github.com/sdamadi/image-classification/blob/main/main.py) the function `init_writer` is used as follows:

```{python}
if args.local_rank == 0:
  writer = init_desc(args, scenario, distributed) 
```

Then, [`writer.py`](https://github.com/sdamadi/image-classification/blob/main/utils/writer.py) uses `scenario.curr_scen_name` to print out the name of the scenario in terminal. 

```
print(f'{scenario.curr_scen_name}\n')
```

Also, upon calling  `init_writer` from [`writer.py`](https://github.com/sdamadi/image-classification/blob/main/utils/writer.py) `scenario.curr_scen_name` is used to create the name of [Tensorboard event](https://github.com/sdamadi/image-classification/blob/main/README/tensorboard.md) as follows:

```
logdir = f'./runs/{args.dataname}/{args.arch}/{scenario.time}'
  if not os.path.exists(logdir): os.makedirs(logdir)
  writer = SummaryWriter(log_dir=logdir, comment = f'_{scenario.curr_scen_name}')
```

In addition, [`writer.py`](https://github.com/sdamadi/image-classification/blob/main/utils/writer.py) instantiate an instance of [`Directories` class](https://github.com/sdamadi/image-classification/blob/main/utils/directories.py) as follows:

```{python}
dirs = Directories(args, scenario.curr_scen_name)
```

Then, the text file that saves all the things that are printed out in the terminal is created by putting these lines of code together in [`utils/directories.py`](https://github.com/sdamadi/image-classification/blob/main/utils/directories.py):


```{python}
self.curr_scen_name = curr_scen_name
self.log_path()
def log_path(self):
  root = 'history/logs'
  self.logpath = f'./{root}/{self.args.dataname}/{self.args.arch}/'\
  f'{self.curr_scen_name}.txt'
```

where `\` is used to break a formatted string in Python when that string is long.


- [`utils/train_validation.py`](https://github.com/sdamadi/image-classification/blob/main/utils/train_validation.py)

Again, when `args.local_rank` is zero, Train and validation methods in [`TraVal class`](https://github.com/sdamadi/image-classification/blob/main/utils/train_validation.py) use `scenario.curr_scen_name` to create Numpy files that save [training and test metrics](https://github.com/sdamadi/image-classification/blob/main/README/training_test_metrics.md). This is done as follows in [`main.py`](https://github.com/sdamadi/image-classification/blob/main/main.py):

```{python}
traval = TraVal(model, train_loader, optimizer,
  criterion, scaler,
  args, validation_loader,
  writer = writer if args.local_rank == 0 else None,
  curr_scen_name = scenario.curr_scen_name if args.local_rank == 0 else None)
```

Then, by instantiation of [`TraVal class`](https://github.com/sdamadi/image-classification/blob/main/utils/train_validation.py) the time of execution (the main name of scenario) is removed from `scenario.curr_scen_name` and only the specific name is used to save [training and test metrics](https://github.com/sdamadi/image-classification/blob/main/README/). The following lines of code do the aforementioned process for saving Numpy variable that saves all the training losses during the trainng.

```{python}
self.curr_scen_name = curr_scen_name
if args.local_rank == 0:
  self.scen_name, self.scen_time = self.time_remover(self.curr_scen_name)
path1 = f'./{root}/{self.args.dataname}/{self.args.arch}/'
self.folder_builder(path = path1, folder_name = self.scen_time)
path = f'{path1}/{self.scen_time}/'
file_name = f'TrainingLoss_{self.scen_name}.npy'
np.save(path + file_name, self.TrainingLoss)
```

- [`utils/outputwriter.py`](https://github.com/sdamadi/image-classification/blob/main/utils/outputwriter.py)

We save the model (neural network) at initialization and after each training epoch. To have these models, we use the specific name as explained above. To do this, in [`main.py`](https://github.com/sdamadi/image-classification/blob/main/main.py) module we first create an instance of [Outputwrite class](https://github.com/sdamadi/image-classification/blob/main/utils/outputwriter.py) by checking whether if the current process is the first process of this scenario as follows:

```{python}
if args.local_rank == 0:
  outputwriter = Outputwriter(model, traval, writer, args,
  scenario.curr_scen_name)
```

Then the time is removed from `scenario.curr_scen_name` while we are constructing an instance of [Outputwrite class](https://github.com/sdamadi/image-classification/blob/main/utils/outputwriter.py) as follows:

```{python}
self.curr_scen_name = curr_scen_name
self.scen_name, self.scen_time = self.time_remover(self.curr_scen_name)
```
Then, the specific scenario name is used to save the model at initialization and after each epoch of training using the following methods of `outputwriter` object as follows:

```{python}
outputwriter.net_params_init()
outputwriter.net_params()
outputwriter.close()
```

All of the above is done when `args.local_rank` is equal to zero.

