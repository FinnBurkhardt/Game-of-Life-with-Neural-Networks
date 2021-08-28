import torch
from torch import nn




class MLP(nn.Module):
    #Multilayer Perceptron
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten =nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(16*16, 256),
            nn.ReLU(),
            nn.Linear(256, 16*16),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.layers(x)
        return x



class CNN(nn.Module):
    #Two Layer Convoltoinal Neural Network
    def __init__(self):
        super(CNN, self).__init__()        
        self.layers = nn.Sequential(
        nn.Conv2d(1,8,kernel_size=3,stride=1,dilation=1,padding=1),#81 params
        nn.ReLU(),
        nn.Conv2d(8,1,kernel_size=1,stride=1,dilation=1,padding=0),#9
        nn.Sigmoid()
        )
    def forward(self, x):
        x = self.layers(x)
        return x


class CNN_residual(nn.Module):
    #Two Layer Covalutional Network using one residual Connection
    def __init__(self):
        super(CNN_residual, self).__init__()        
        self.layers = nn.Sequential(
        nn.Conv2d(1,8,kernel_size=3,stride=1,dilation=1,padding=1),#81 params
        nn.ReLU(),
        nn.Conv2d(8,1,kernel_size=1,stride=1,dilation=1,padding=0),#9
        nn.Tanh()
        )
    def forward(self, x):
        out = self.layers(x)
        x = x+out
        return x


class MLP_residual(nn.Module):
    #Multilayer Perceptron with one residual Connection
    def __init__(self):
        super(MLP_residual, self).__init__()
        self.flatten =nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(16*16, 256),
            nn.ReLU(),
            nn.Linear(256, 16*16),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.flatten(x)
        out = self.layers(x)
        x = x+out
        return x