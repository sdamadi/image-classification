import numpy as np

class LR(object):

  def __init__(self, lr_policy, base_lr, warmup_length, epochs, initial_stage, stages, 
              lowest_lr, scale_coslr, exp_coslr, 
               normal_exp_scale, lr_gamma, lr_steps):
      
      self.lr_policy = lr_policy
      self.base_lr = base_lr
      self.warmup_length = warmup_length
      self.epochs = epochs
      self.stages = stages
      self.initial_stage = initial_stage
      self.lowest_lr = lowest_lr
      self.scale_coslr = scale_coslr
      self.exp_coslr = exp_coslr
      self.normal_exp_scale = normal_exp_scale
      self.lr_gamma = lr_gamma
      self.lr_steps = lr_steps
       

  def warmup_lr(self, epoch):
      self.lr =  self.base_lr * (epoch + 1) / self.warmup_length

  def apply_lr(self, epoch, stage):

    adjusted_epoch =  stage*self.epochs +  epoch
    e = adjusted_epoch - self.warmup_length
    es = (self.epochs*(self.stages + self.initial_stage+1)) - self.warmup_length

    if adjusted_epoch < self.warmup_length:
        self.warmup_lr(adjusted_epoch)
    else:
      if self.lr_policy == 'constant_lr':
          self.lr = self.base_lr

      elif self.lr_policy == 'cosine_lr':
        lr_init_stage = self.base_lr*(100-self.initial_stage)/100

        if self.initial_stage != 0:
          if e < self.initial_stage:
            a = 0.5*(self.base_lr + lr_init_stage)
            b = 0.5*(self.base_lr - lr_init_stage)
            self.lr = a +  b*np.cos(np.pi * e / self.initial_stage)
          else: 
            a = 0.5*(lr_init_stage + self.lowest_lr)
            b = 0.5*(lr_init_stage - self.lowest_lr)
            self.lr = a + b*np.cos(np.pi * (e-self.initial_stage) / (self.scale_coslr*(es-self.initial_stage)) )**self.exp_coslr
        else:
          self.lr = 0.5 * (1 + np.cos(np.pi * e / (self.scale_coslr*es))**self.exp_coslr) * self.base_lr
      
      elif self.lr_policy == 'normal_lr':
        lr_init_stage = self.base_lr*(100-self.initial_stage)/100

        if self.initial_stage != 0:
          if e < self.initial_stage:
            a = 0.5*(self.base_lr + lr_init_stage)
            b = 0.5*(self.base_lr - lr_init_stage)
            self.lr = a +  b*np.cos(np.pi * e / self.initial_stage)
          else:
            
            self.lr = lr_init_stage*np.exp((-(e-self.initial_stage)**2)/self.normal_exp_scale) + self.lowest_lr

        else: 
          self.lr = lr_init_stage*np.exp((-(e-self.initial_stage)**2)/self.normal_exp_scale) + self.lowest_lr
      elif self.lr_policy == 'multistep_lr':
        n = len(self.lr_steps)
        lr_steps = [0] + self.lr_steps + [es]
        for j in range(n+1):
          if lr_steps[j] <= e <lr_steps[j+1]:
            self.lr = self.base_lr * (self.lr_gamma ** (j))
            # print(f'e: {e:3.0f} | lr: {self.lr:.6f}')
            break
    return self.lr