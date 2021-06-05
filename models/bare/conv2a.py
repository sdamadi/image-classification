import torch
import torch.nn as nn 

class Conv2A(nn.Module):
  def __init__(self, in_ch, imgsz, num_classes=10):
    super(Conv2A, self).__init__()
    self.conv1 = nn.Conv2d(in_ch, 64, kernel_size=(3, 3), stride=1, padding=1, bias=False)
    self.conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1, bias=False)
    self.maxpool = nn.MaxPool2d(kernel_size=2)
    self.relu = nn.ReLU(inplace=True)
    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    self.fc = nn.Linear(64, num_classes)

  def forward(self, x):
    x = self.relu(self.conv1(x))
    x = self.relu(self.conv2(x))
    x = self.maxpool(x)
    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    x = self.fc(x)
    return x