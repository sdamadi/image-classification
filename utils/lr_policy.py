import numpy as np

class LR(object):

  def __init__(self, lr_policy, base_lr, 
              warmup_length, initial_epoch, epochs, 
              lowest_lr, scale_coslr, exp_coslr, 
              normal_exp_scale, lr_gamma, lr_steps):
      
      self.lr_policy = lr_policy
      self.base_lr = base_lr
      self.warmup_length = warmup_length
      self.initial_epoch = initial_epoch
      self.epochs = epochs
      self.lowest_lr = lowest_lr
      self.scale_coslr = scale_coslr
      self.exp_coslr = exp_coslr
      self.normal_exp_scale = normal_exp_scale
      self.lr_gamma = lr_gamma
      self.lr_steps = lr_steps
       

  def warmup_lr(self, epoch):
      self.lr =  self.base_lr * (epoch + 1) / self.warmup_length
  
  def apply_lr(self, epoch):
    if epoch < self.warmup_length:
        self.warmup_lr(epoch)
    else:

      # translated epoch to the right by ignoring number of warmups
      e = epoch - self.warmup_length
      # the denominator of the angle in cosine
      es = self.epochs - self.warmup_length

      if self.lr_policy == 'constant_lr':
          self.lr = self.base_lr

      elif self.lr_policy == 'cosine_lr':
        lr_init_epoch = self.base_lr*(100-self.initial_epoch)/100

        if self.initial_epoch != 0:
          if e < self.initial_epoch:
            a = 0.5*(self.base_lr + lr_init_epoch)
            b = 0.5*(self.base_lr - lr_init_epoch)
            self.lr = a +  b*np.cos(np.pi * e / self.initial_epoch)
          else: 
            a = 0.5*(lr_init_epoch + self.lowest_lr)
            b = 0.5*(lr_init_epoch - self.lowest_lr)
            self.lr = a + b*np.cos(np.pi * (e-self.initial_epoch) / (self.scale_coslr*(es-self.initial_epoch)) )**self.exp_coslr
        else:
          self.lr = 0.5 * (1 + np.cos(np.pi * e / (self.scale_coslr*es))**self.exp_coslr) * self.base_lr
      
      
      elif self.lr_policy == 'normal_lr':
        # self.lr = self.base_lr
        
        lr_init_epoch = self.base_lr*(100-self.initial_epoch)/100
        self.lr = lr_init_epoch*np.exp((-(e-self.initial_epoch)**2)/self.normal_exp_scale) + self.lowest_lr

        # if self.initial_epoch != 0:
        #   if e < self.initial_epoch:
        #     a = 0.5*(self.base_lr + lr_init_epoch)
        #     b = 0.5*(self.base_lr - lr_init_epoch)
        #     self.lr = a +  b*np.cos(np.pi * e / self.initial_epoch)
        #   else:
        #     print('we are here!')
            
            
      
      elif self.lr_policy == 'multistep_lr':
          self.lr = self.base_lr
    
    
    
    return self.lr