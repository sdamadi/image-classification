import torch.nn as nn 
from .l1 import *

class STRFC(nn.Module):
  def __init__(self, str_args, imgsz=28, num_classes=10, *args, **kwargs):
      super(STRFC, self).__init__(*args, **kwargs)
      self.imgsz = imgsz
      self.strfc1 = STRConv(in_channels=imgsz*imgsz, out_channels=300, kernel_size=1, str_args=str_args)
      self.strfc2 = STRConv(in_channels=300, out_channels=100, kernel_size=1, str_args=str_args)
      self.strfc3 = STRConv(in_channels=100, out_channels=num_classes, kernel_size=1, str_args=str_args)
      self.relu = nn.ReLU(inplace=True)

  def forward(self, x):
      x = x.view(-1, self.imgsz*self.imgsz, 1, 1)
      x = self.relu(self.strfc1(x))
      x = self.relu(self.strfc2(x))
      x = self.strfc3(x)
      x = x.view(x.shape[0], -1)
      return x