import torch
import torch.nn as nn

class Conv4(nn.Module):
  def __init__(self, in_ch, imgsz, num_classes=10):
    super(Conv4, self).__init__()
    self.conv1 = nn.Conv2d(in_ch, 64, kernel_size=(3, 3), stride=1, padding=1)
    self.conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1)
    self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=1)
    self.conv4 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1, padding=1)
    self.maxpool = nn.MaxPool2d(kernel_size=2)
    self.relu = nn.ReLU(inplace=True)
    self.fc1 = nn.Linear(128*(imgsz//4)*(imgsz//4), 256)
    self.fc2 = nn.Linear(256, 256)
    self.fc3 = nn.Linear(256, num_classes)

  def forward(self, x):
    x = self.relu(self.conv1(x))
    x = self.relu(self.conv2(x))
    x = self.maxpool(x)
    x = self.relu(self.conv3(x))
    x = self.relu(self.conv4(x))
    x = self.maxpool(x)
    x = x.view( x.size(0), -1)
    x = self.relu(self.fc1(x))
    x = self.relu(self.fc2(x))
    x = self.fc3(x)
    return x