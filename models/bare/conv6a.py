import torch
import torch.nn as nn

class Conv6A(nn.Module):
  def __init__(self, in_ch, imgsz, num_classes=10):
    super(Conv6A, self).__init__()
    self.conv1 = nn.Conv2d(in_ch, 64, kernel_size=(3, 3), stride=1, padding=1)
    self.conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1)
    self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=1)
    self.conv4 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1, padding=1)
    self.conv5 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=1, padding=1)
    self.conv6 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=1, padding=1)
    self.maxpool = nn.MaxPool2d(kernel_size=2)
    self.relu = nn.ReLU(inplace=True)
    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    self.fc = nn.Linear(256, num_classes)

  def forward(self, x):
    x = self.relu(self.conv1(x))
    x = self.relu(self.conv2(x))
    x = self.maxpool(x)
    x = self.relu(self.conv3(x))
    x = self.relu(self.conv4(x))
    x = self.maxpool(x)
    x = self.relu(self.conv5(x))
    x = self.relu(self.conv6(x))
    x = self.maxpool(x)
    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    x = self.fc(x)
    return x