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


def get_sparse_init(model, dataname, arch, prepruned_model, prepruned_scen,
                    nonpruned_percent, mask_stage,
                    local_rank, quantize, bias):
  main_path = f'./history/saved/{dataname}/{arch}/'
  pruned_path = main_path  + 'saved_at_each_stage_of_pruning/' + prepruned_scen
  initialization_path = main_path  + 'saved_initialization/' + prepruned_scen

  for i in os.listdir(pruned_path):
    if f'{nonpruned_percent}' in i and f'{mask_stage}' in i:
      pruned_path = pruned_path + '/' + i
      if local_rank == 0:
        print('========> The following is the pruned model being read for training:\n')
        print(f'{i}\n')
      break
    
  d = torch.load(pruned_path, map_location=torch.device('cpu'))
  d_model = d.copy()
  for k, v in d_model.items():
    d[k.replace('module.', '')] = d.pop(k)
  del d_model
  model.load_state_dict(d)
  del d

  mask = [ torch.where(param.data != torch.tensor(0.).to(param.device), 
          torch.tensor(1.).to(param.device) , torch.tensor(0.).to(param.device)) for name, param in model.named_parameters()]

  if prepruned_model and not quantize:
    initialization_path = initialization_path + '/' + next(os.walk(initialization_path))[2][0]
    d = torch.load(initialization_path, map_location=torch.device('cpu'))
    d_model = d.copy()
    for k, v in d_model.items():
      d[k.replace('module.', '')] = d.pop(k)
    del d_model
    model.load_state_dict(d)
    del d

  if quantize: 
    net_quantizer(model, bias)

  for j, p in enumerate(model.parameters()):
      p.data = p.data*mask[j].to(p.device)
  del mask

  return model

def net_quantizer(model, bias=False, reset_bias=True, reset_bn=True):    
  for name, module in model.named_modules():
    with torch.no_grad():
      if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
      #layer-wise quantization that calculates the centroids on the fly
        p = module.weight.data
        c_plus = p[p>0].mean()
        c_minus = p[p<0].mean()
        p[p>0] = c_plus
        p[p<0] = c_minus
        module.weight.data = p
        if module.bias is not None and bias:
          p = module.bias.data
          c_plus = p[p>0].mean()
          c_minus = p[p<0].mean()
          p[p>0] = c_plus
          p[p<0] = c_minus
          module.bias.data = p
        elif module.bias is not None and not bias and reset_bias:
          nn.init.zeros_(module.bias)
      elif isinstance(module, nn.BatchNorm2d) and reset_bn:
        nn.init.ones_(module.weight)
        if module.bias is not None and not bias and reset_bias:
          nn.init.zeros_(module.bias)