import torch
import torch.nn as nn 
from torch.nn import functional as F

class STRConv(nn.Conv2d):
  def __init__(self, str_args, *args, **kwargs):
    super(STRConv, self).__init__(*args, **kwargs)
    self.args = str_args
    if self.args.str_activation == 'relu':
      self.activation = torch.relu 
    self.nonlinear_function = self.args.str_nonlinear
    th_init_type = self.args.init_threshold_type
    th_init_value = torch.tensor(self.args.init_threshold)

    if self.nonlinear_function == 'sigmoid':
        self.f = torch.sigmoid
        self.Threshold = nn.Parameter(self.th_initialization(th_init_type, th_init_value))
    elif self.nonlinear_function == 'none':
      self.f = nn.Identity
      self.Threshold = nn.Parameter(self.th_initialization(th_init_type, th_init_value))

  def forward(self, x):
    sparseWeight = self.L1norm_sol(self.weight, self.Threshold, self.activation, self.f).to(self.weight.device)
    x = F.conv2d(x, sparseWeight, self.bias, self.stride, 
                  self.padding, self.dilation, self.groups)    
    return x

  def th_initialization(self, th_init_type, th_init_value):
      if th_init_type == "constant":
          return th_init_value*torch.ones(1,1)

  def L1norm_sol(self, w, th, activation=torch.relu, g=torch.sigmoid):
    return torch.sign(w)*activation(torch.abs(w)-g(th))

  def getSparsity(self):
    sparseWeight = L1norm_sol(self.weight, self.Threshold,  self.activation, self.f)
    temp = sparseWeight.detach().cpu()
    temp[temp!=0] = 1
    return (100 - temp.mean().item()*100), temp.numel(), self.f(self.Threshold).item()
