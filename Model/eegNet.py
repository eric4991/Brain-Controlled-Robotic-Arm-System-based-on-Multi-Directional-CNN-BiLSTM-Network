import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchsummary import summary

class eegNet(nn.Module):
    def __init__(self,outputSize):
        self.outputSize=outputSize
        super(eegNet,self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(1,20,kernel_size=[1,50],padding=0),
            nn.BatchNorm2d(20),
            nn.Conv2d(20,40,kernel_size=[20,1],padding=0,groups=20),
            nn.BatchNorm2d(40),
            nn.ELU(True),
            nn.AvgPool2d([1,4],stride=[1,4],padding=0),
            nn.Dropout2d(p=0.5),
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(40,80,kernel_size=[1,12],padding=0),
            nn.BatchNorm2d(80),
            nn.ELU(True),
            nn.AvgPool2d([1,8],stride=[1,8],padding=0),
            nn.Dropout2d(p=0.5)
        )
        self.fc=nn.Linear(720,self.outputSize)
    def forward(self,x):
        out=self.conv1(x)
        out=self.conv2(out)
        out=out.view(out.size(0),-1)
        out=self.fc(out)
        out=out.view(out.size(0),3,-1)
        return out

def veri():
    net=eegNet(903)
    x=torch.randn(1,1,20,401)
    y=net(x)
    print(y.size())

#veri()