import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchsummary import summary

class eegNet300(nn.Module):
    def __init__(self,outputSize):
        self.outputSize=outputSize
        super(eegNet300,self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(1,8,kernel_size=[1,50],padding=0),
            nn.BatchNorm2d(8),
            nn.Conv2d(8,16,kernel_size=[20,1],padding=0,groups=8),
            nn.BatchNorm2d(16),
            nn.ELU(True),
            nn.AvgPool2d([1,4],stride=[1,4],padding=0),
            nn.Dropout2d(p=0.5),
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(16,32,kernel_size=[1,12],padding=0),
            nn.BatchNorm2d(32),
            nn.ELU(True),
            nn.AvgPool2d([1,8],stride=[1,8],padding=0),
            nn.Dropout2d(p=0.5)
        )
        self.fc=nn.Linear(480,self.outputSize)
    def forward(self,x):
        out=self.conv1(x)
        out=self.conv2(out)
        print(out.size())
        out=out.view(out.size(0),-1)
        print(out.size())
        out=self.fc(out)
        out=out.view(out.size(0),3,-1)
        return out

def veri():
    net=eegNet300(903)
    x=torch.randn(1,1,24,301)
    y=net(x)
    print(y.size())

#veri()