import os
from os import listdir

import torch
import torch.nn as nn
import torch.optim as optim
from models.str_l1.l1 import STRConv


class Poda(object):
    
  def __init__(self, model, args):
    self.model = model
    self.args = args
    self.args.prepruned_model = args.prepruned_model
    self.global_threshold = 0
    self.args.prepruned_model = args.prepruned_model
    self.initial_mask_list(self.model)
    self.sparsity = list()
    self.Prune_Percent = []
    self.Learning_Rate = []
    
      
  def initial_mask_list(self, model):
    # building the mask list and filling that out
    if not self.args.prepruned_model and self.args.pruning_strategy in {'asni', 'lottery'}:
        self.mask = [ torch.ones_like( param.data ) for name, param in model.named_parameters()]
    elif self.args.prepruned_model:
        self.mask = [ torch.where(param.data != torch.tensor(0.).to(param.device), 
                      torch.tensor(1.).to(param.device) , torch.tensor(0.).to(param.device)) for name, param in model.named_parameters()]
        
  def subnetwork_picker(self, model, percent, local_prune, prune_bn, prune_bias):
    with torch.no_grad():
      # global pruning
      if (not local_prune) and percent != 0 and (not self.args.prepruned_model):
        modules_mask = []
        for name, module in model.named_modules():
          if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            modules_mask.append(module.weight.data.flatten())
            if module.bias is not None and prune_bias:
              modules_mask.append(module.bias.data.flatten())
        torch_mask = torch.cat(modules_mask, dim = 0)
        global_threshold = self.threshold(torch_mask, percent)
        self.global_threshold = global_threshold

        for name, module in model.named_modules():
          if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            module.weight.data = self.truncate(module.weight.data, global_threshold)
            if module.bias is not None and prune_bias:
              module.bias.data = self.truncate(module.bias.data, global_threshold)

      # Local pruning
      elif local_prune and self.percent!=0 and (not self.prepruned_model):
        for name, module in model.named_modules():
          if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            module.weight.data = self.project(module.weight.data, percent)
            if module.bias is not None and prune_bias:
              module.bias.data = self.project(module.bias.data, percent)


      self.mask = [ torch.where(param.data != torch.tensor(0.).to(param.device),
                        torch.tensor(1.).to(param.device) , torch.tensor(0.).to(param.device)) for name, param in model.named_parameters()]
            
        
  def threshold(self, p, percent):
    pvec = p.detach().clone().flatten()
    palive = pvec[torch.nonzero(pvec)].flatten()
    k = int( palive.numel()*percent/100 )
    t = torch.topk(torch.abs(palive), k, largest = False)[0].max()
    return t

  def project(self, p, percent):
    t = self.threshold(p, percent)
    p_proj = self.truncate(p, t)
    return p_proj

  def truncate(self, p, thshd):
    pcopy = p.detach().clone()
    p_down = torch.where(torch.abs(pcopy) > thshd, pcopy, torch.tensor(0.).to(thshd.device))
    return p_down

  
  def network_status(self, stage, total_params, nz_params, report_nwtwork):
    z_params = total_params - nz_params
    eta = nz_params/total_params

    if stage>= self.args.initial_stage and not self.args.prepruned_model:
        if self.args.pruning_strategy in 'asni' or (self.args.pruning_strategy == 'lottery' and self.args.percent !=0):
          prune_stg = stage - self.args.initial_stage
        else:
          prune_stg = 0
    elif stage>= self.args.initial_stage and self.args.prepruned_model:
      prune_stg = 0
    
    if (self.args.pruning_strategy in 'asni' or (self.args.pruning_strategy == 'lottery' and self.args.percent !=0) ) and not self.args.prepruned_model:
      prune_stgs = self.args.stages
    elif self.args.prepruned_model:
      prune_stgs = 0
    else: 
      prune_stgs = 0

    if report_nwtwork:
        print(f'\nStg: [{stage :2}/{self.args.stages + self.args.initial_stage:2}] | '\
        f'Pruning Stg: [{prune_stg:2}/{prune_stgs :2}] | '\
        f'Nonz: {(nz_params)} | Pruned: {(z_params)} | Total: {total_params} | '\
        f'Comp: {total_params/nz_params:5.2f}x  ({100 * (1-eta):4.2f}% pruned)\n')

  def layer_status(self, name, nz_count, total_params, size, report_layer):
    if report_layer:
      print(f'{name:32} | nonzeros = {(nz_count):7} / {(total_params):7} '\
          f'({100 * nz_count / total_params:6.2f}%) '\
          f'| total_pruned = {(total_params - nz_count) :7} | shape = {size}')

  def params_status(self, stage, report_layer=True, report_network = True):
    self.status = list()
    nonzero = 0
    total = 0
    s = list()
    if self.args.pruning_strategy in {'asni', 'lottery', 'quantize', 'prepruned'}:
      with torch.no_grad():
        for l_type ,(name, p) in enumerate(self.model.named_parameters()):
          nz_count = torch.count_nonzero(p.data)
          total_params = p.data.numel()
          nonzero += nz_count
          total += total_params
          s.append(100 * nz_count / total_params)
          new_name = name.replace('.','_').replace('module_', '')
          size_ls = list(p.size())
          size_str = str(size_ls).replace('[','(').replace(']', ')')
          self.layer_status(new_name, nz_count, total_params, size_str, report_layer)
        self.sparsity.append(s)
        self.eta = nonzero/total
        self.network_status(stage, total, nonzero, report_network)
    
    elif self.args.pruning_strategy == 'str':
      
      with torch.no_grad():
        for name, module in self.model.named_modules():
          if isinstance(module, STRConv):
            mask = torch.where(torch.abs(module.weight.data) >= torch.sigmoid(module.Threshold), 1, 0)
            nz_count = mask.sum()
            total_params = module.weight.data.numel()
            nonzero += nz_count
            total += total_params
            s.append(100 * nz_count / total_params)
            new_name = name.replace('.','_').replace('module_', '') + '_weight'
            size_ls = list(module.weight.data.size())
            size_str = str(size_ls).replace('[','(').replace(']', ')') 
            self.layer_status(new_name, nz_count, total_params, size_str, report_layer)
            if module.bias is not None:
              mask = torch.where(torch.abs(module.bias.data) >= torch.sigmoid(module.Threshold), 1, 0)
              nz_count = mask.sum()
              total_params = module.bias.data.numel()
              nonzero += nz_count
              total += total_params
              s.append(100 * nz_count / total_params)
              new_name = name.replace('.','_').replace('module_', '') + '_bias'
              size_ls = list(module.bias.data.size())
              size_str = str(size_ls).replace('[','(').replace(']', ')') 
              self.layer_status(new_name, nz_count, total_params, size_str, report_layer)

        self.sparsity.append(s)
        print(total)
        self.eta = nonzero/total 
        self.network_status(stage, total, nonzero, report_network)
    self.pruned = round(100*(1-self.eta).item(), 2)

# def get_sparse_init(model, dataname, arch, prepruned_model, prepruned_scen,
#                     nonpruned_percent, local_rank, quantize, bias):
#   main_path = f'./history/saved/{dataname}/{arch}/'
#   pruned_path = main_path  + 'saved_at_each_stage_of_pruning/' + prepruned_scen
#   initialization_path = main_path  + 'saved_initialization/' + prepruned_scen

#   for i in os.listdir(pruned_path):
#     if f'{nonpruned_percent}' in i:
#       pruned_path = pruned_path + '/' + i
#       if local_rank == 0:
#         print('========> The following is pruned model being read for training:\n')
#         print(f'{i}\n')
#       break
    
#   d = torch.load(pruned_path)
#   d_model = d.copy()
#   for k, v in d_model.items():
#     d[k.replace('module.', '')] = d.pop(k)
#   del d_model
#   model.load_state_dict(d)
#   del d

#   mask = [ torch.where(param.data != torch.tensor(0.).to(param.device), 
#           torch.tensor(1.).to(param.device) , torch.tensor(0.).to(param.device)) for name, param in model.named_parameters()]

#   if prepruned_model and not quantize:
#     initialization_path = initialization_path + '/' + next(os.walk(initialization_path))[2][0]
#     d = torch.load(initialization_path)
#     d_model = d.copy()
#     for k, v in d_model.items():
#       d[k.replace('module.', '')] = d.pop(k)
#     del d_model
#     model.load_state_dict(d)
#     del d

#   if quantize: net_quantizer(model, bias)

#   for j, p in enumerate(model.parameters()):
#       p.data = p.data*mask[j].to(p.device)
#   del mask

#   return model
  

# def net_quantizer(model, bias):    
#   for name, module in model.named_modules():
#     if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
#       #layer-wise quantization that calculates the centroids on the fly
#       with torch.no_grad():
#         p = module.weight.data
#         c_plus = p[p>0].mean()
#         c_minus = p[p<0].mean()
#         p[p>0] = c_plus
#         p[p<0] = c_minus
#         module.weight.data = p
#         if module.bias.data is not None and bias:
#           p = module.bias.data
#           c_plus = p[p>0].mean()
#           c_minus = p[p<0].mean()
#           p[p>0] = c_plus
#           p[p<0] = c_minus
#           module.bias.data = p
  
    
  
  
