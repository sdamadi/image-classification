import torch.nn as nn 

class FC(nn.Module):
  def __init__(self, imgsz=28, num_classes=10):
      super(FC, self).__init__()
      self.imgsz = imgsz
      self.fc1 = nn.Linear(imgsz*imgsz, 300)
      self.fc2 = nn.Linear(300, 100)
      self.fc3 = nn.Linear(100, num_classes)
      self.relu = nn.ReLU(inplace=True)

  def forward(self, x):
      x = x.view(-1, self.imgsz*self.imgsz)
      x = self.relu(self.fc1(x))
      x = self.relu(self.fc2(x))
      x = self.fc3(x)
      
      return x