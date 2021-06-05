import torch
import torch.nn as nn
from .l1 import *
from argparse import Namespace


cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class STRVGG(nn.Module):
    def __init__(self, str_args, vgg_name, num_classes, **kwargs):
        super(STRVGG, self).__init__()
        self.str_args = str_args
        self.conv = self._make_layers(cfg[vgg_name])
        self.fc = STRConv(in_channels=512, out_channels=num_classes,
                            kernel_size=1, str_args=self.str_args
                            )
    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), -1, 1, 1)
        out = self.fc(out)
        out = out.view(out.shape[0], -1)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [STRConv(in_channels=in_channels, out_channels=x, kernel_size=3, padding=1, str_args=self.str_args),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)