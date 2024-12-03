import torch.nn as nn
import torch

class DepthModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        ## REMOVE ALL OF THIS AND MAKE A DIFFERENT ARCHITECTURE ##
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        
        ## REMOVE ALL OF THIS AND MAKE A DIFFERENT ARCHITECTURE ##
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.relu(self.fc1(x))
        return self.fc2(x)