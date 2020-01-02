import torch
import torch.nn as nn
import torch.nn.functional as F


class EstimationModelByOneHot(nn.Module):
    def __init__(self):
        super(EstimationModelByOneHot, self).__init__()
        # input: 14(cards)*4(players) + 4(players) + 4(True or False) = 64
        self.net = nn.Sequential(
            nn.Linear(64, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 14)
        )
        #pytorchのcrossentropylossにsoftmaxが入っているので、抜いておきました。
        self.softmax = nn.Softmax()
    
    def forward(self, x):
        x = self.net(x)
        return x
    
    def estimate(self, x):
        x = self.net(x)
        x = self.softmax(x)
        return x

class EstimationModelByContinuous(nn.Module):

    def __init__(self):
        super(EstimationModelByContinuous, self).__init__()
        # input: 5(B)*4(players) + 4(players) + 4(True or False) = 28
        self.net = nn.Sequential(
            nn.Linear(28, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200,200),
            nn.ReLU(),
            nn.Linear(200,200),
            nn.ReLU(),
            nn.Linear(200, 14)
        )
        self.softmax = nn.Softmax()
    
    def forward(self, x):
        y = self.net(x)
        return y
    
    def estimate(self, x):
        x = self.net(x)
        x = self.softmax(x)
        return x
