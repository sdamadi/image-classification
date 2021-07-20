import torch
import torch.nn as nn
import torch.nn.init as init
import os

def network_init(net, init_policy, init_mode, init_nonlinearity, init_bias):
  for module in net.modules():
    weights_init(module, init_policy, init_mode, init_nonlinearity, init_bias)

def weights_init(m, init_policy, init_mode, init_nonlinearity, init_bias):
  
  if init_policy == 'xaviern':
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
      nn.init.xavier_normal_(m.weight)
      if m.bias is not None and init_bias == 'zero':
        torch.nn.init.zeros_(m.bias)
  elif init_policy == 'xavieru':
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
      nn.init.xavier_uniform_(m.weight)
      if m.bias is not None and init_bias == 'zero':
        torch.nn.init.zeros_(m.bias)
  elif init_policy == 'kaimingn':
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
      nn.init.kaiming_normal_(m.weight, mode = init_mode, nonlinearity = init_nonlinearity)
      if m.bias is not None and init_bias == 'zero':
        torch.nn.init.zeros_(m.bias)
  elif init_policy == 'kaimingn':
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
      nn.init.kaiming_normal_(m.weight, mode = init_mode, nonlinearity = init_nonlinearity)
      if m.bias is not None and init_bias == 'zero':
        torch.nn.init.zeros_(m.bias)
