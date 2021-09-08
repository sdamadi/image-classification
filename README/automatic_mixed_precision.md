### Pytorch Automatic Mixed Precision (AMP)

This repository utilizes [native Pytorch Automatic Mixed Precision (AMP)](https://pytorch.org/blog/accelerating-training-on-nvidia-gpus-with-pytorch-automatic-mixed-precision/) package for fast training. 

**Remark1:**
Note that [Apex module developed by Nvidia](https://github.com/NVIDIA/apex) does the same thing but we rather use the native Pytorch module to make the code more compact. 

**Remark2:**
The advantage of using the Automatic Mixed Precision module is that the speed of training gets boosted twice, at least. Also, be mindful that the Automatic Mixed Precision is only used in the training process.

### How to use it?
We mainly follow [Typical Mixed Precision Training documention for Pytorch](https://pytorch.org/docs/stable/notes/amp_examples.html#typical-mixed-precision-training). However, in our code there are two modules dealing with the Automatic Mixed Precision as follows: 
- `main.py`
- `utils/train_validation.py`.

#### `main.py`
1. To get started with the Automatic Mixed Precision one needs to create `GradScaler` instance in `main.py` file before the beginning of the training.
```{python}
scaler = torch.cuda.amp.GradScaler()
```

2. Again, in `main.py` pass the `scaler` instance to `TraVal` class to create an instance of taining and validation.

```{python}
traval = TraVal(model, train_loader, optimizer,
                 criterion, scaler,
                 args, validation_loader,
                 writer = writer if args.local_rank == 0 else None,
                 curr_scen_name = scenario.curr_scen_name if args.local_rank == 0 else None)
```
#### `utils/train_validation.py`

3. In `train_validation.py` import `autocast` class from `torch.cuda.amp` using the following:

```
from torch.cuda.amp import autocast
``` 

4. In training process only forward pass with autocasting are recommended. Do not use backward passes. Therefore, `autocast` only wraps the forward pass(es) of the network, including the loss computation(s) as the followning:

```{python}
with autocast():
  output = self.model(input)
  loss = self.criterion(output, target) 
```

5. Scale the loss because training with  the Automatic Mixed Precision requires that the loss is scaled in order to prevent the gradients from underflow.

```{python}
self.scaler.scale(loss).backward()
```

6. Finally, `self.scaler.step()` unscales the gradients of the optimizer's assigned parameters. If these gradients do not contain `inf`s or `NaN`s, `self.optimizer.step()` is then called. Otherwise, `self.optimizer.step()` is skipped.

7. Updates the scale for next iteration.
```{python}
self.scaler.update()
```

