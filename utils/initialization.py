import torch
import torch.nn as nn
import torch.nn.init as init
import os


def network_init(model, args):  
  # ===== Dense Initialization ===== #
  dense_init_policy = args.dense_init_policy
  kaiming_mode = args.kaiming_mode
  kaiming_nonlinearity = args.kaiming_nonlinearity
  dense_init_bias = args.dense_init_bias
  if args.dense_init_policy in {'kaimingn', 'kaimingu', 'xaviern', 'xavieru'}:
    dense_init(model,
            dense_init_policy,
            kaiming_mode, 
            kaiming_nonlinearity,
            dense_init_bias)
    

def dense_init(model, init_policy, init_mode, init_nonlinearity, init_bias):
  for module in model.modules():
    weights_init(module, init_policy, init_mode, init_nonlinearity, init_bias)

def weights_init(m, init_policy, init_mode, init_nonlinearity, init_bias):
  
  if init_policy == 'xaviern':
    if isinstance(m, (nn.Linear, nn.Conv2d)):
      nn.init.xavier_normal_(m.weight)
      if m.bias is not None and init_bias == 'zero':
        torch.nn.init.zeros_(m.bias)
  elif init_policy == 'xavieru':
    if isinstance(m, (nn.Linear, nn.Conv2d)):
      nn.init.xavier_uniform_(m.weight)
      if m.bias is not None and init_bias == 'zero':
        torch.nn.init.zeros_(m.bias)
  elif init_policy == 'kaimingn':
    if isinstance(m, (nn.Linear, nn.Conv2d)):
      nn.init.kaiming_normal_(m.weight, mode = init_mode, nonlinearity = init_nonlinearity)
      if m.bias is not None and init_bias == 'zero':
        torch.nn.init.zeros_(m.bias)
  elif init_policy == 'kaimingn':
    if isinstance(m, (nn.Linear, nn.Conv2d)):
      nn.init.kaiming_normal_(m.weight, mode = init_mode, nonlinearity = init_nonlinearity)
      if m.bias is not None and init_bias == 'zero':
        torch.nn.init.zeros_(m.bias)
