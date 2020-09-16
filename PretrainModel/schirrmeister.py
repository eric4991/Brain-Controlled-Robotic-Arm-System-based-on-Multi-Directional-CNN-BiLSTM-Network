import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchsummary import summary

class schirrmeister(nn.Module):
    def __init__(self,outputSize):
        self.outputSize=outputSize
        super(schirrmeister,self).__init__()
        self.preLayers=nn.Sequential(
            nn.Conv2d(1,25,kernel_size=[1,10],padding=0),
            nn.BatchNorm2d(25),
            nn.ELU(True),
        )
        self.spatial=nn.Sequential(
            nn.Conv2d(25,25,kernel_size=[20,1],padding=0),
            nn.BatchNorm2d(25),
            nn.ELU(True),
        )
        self.maxpool1=nn.AvgPool2d([1,3],stride=[1,3],padding=0)
        self.conv1=nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(25,50,kernel_size=[1,10],padding=0),
            nn.BatchNorm2d(50),
            nn.ELU(True),
        )
        self.maxpool2=nn.AvgPool2d([1,3],stride=[1,3],padding=0)
        self.conv2=nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(50,100,kernel_size=[1,10],padding=0),
            nn.BatchNorm2d(100),
            nn.ELU(True),
        )
        self.maxpool3=nn.AvgPool2d([1,3],stride=[1,3],padding=0)
        self.conv3=nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(100,200,kernel_size=[1,10],padding=0),
            nn.ELU(True),
        )
        self.avgpool=nn.AvgPool2d(1,padding=0)
        self.linear=nn.Linear(200,self.outputSize)
    def forward(self,x):
        out=self.preLayers(x)
        out=self.spatial(out)
        out=self.maxpool1(out)
        out=self.conv1(out)
        out=self.maxpool2(out)
        out=self.conv2(out)
        out=self.maxpool3(out)
        out=self.conv3(out)
        out=self.avgpool(out)
        out=out.view(out.size(0),-1)
        out=self.linear(out)
        return out

class schirrmeister1(nn.Module):
    def __init__(self,outputSize):
        self.outputSize=outputSize
        super(schirrmeister1,self).__init__()
        self.preLayers=nn.Sequential(
            nn.Conv2d(1,25,kernel_size=[1,10],padding=0),
            nn.BatchNorm2d(25),
            nn.ELU(),
        )
        self.spatial=nn.Sequential(
            nn.Conv2d(25,25,kernel_size=[20,1],padding=0),
            nn.BatchNorm2d(25),
            nn.ELU(),
        )
        self.maxpool1=nn.AvgPool2d([1,3],stride=[1,3],padding=0)
        self.conv1=nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(25,50,kernel_size=[1,10],padding=0),
            nn.BatchNorm2d(50),
            nn.ELU(),
        )
        self.maxpool2=nn.AvgPool2d([1,3],stride=[1,3],padding=0)
        self.conv2=nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(50,100,kernel_size=[1,10],padding=0),
            nn.BatchNorm2d(100),
            nn.ELU(),
        )
        self.maxpool3=nn.AvgPool2d([1,3],stride=[1,3],padding=0)
        self.conv3=nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(100,200,kernel_size=[1,10],padding=0),
            nn.ELU(),
        )
        self.avgpool=nn.AvgPool2d(1,padding=0)
        self.linear1=nn.Sequential(
            nn.Linear(200,200),
            nn.ReLU(True),
            nn.Dropout(),
        )
        self.linear2=nn.Linear(200,self.outputSize)
    def forward(self,x):
        out=self.preLayers(x)
        out=self.spatial(out)
        out=self.maxpool1(out)
        out=self.conv1(out)
        out=self.maxpool2(out)
        out=self.conv2(out)
        out=self.maxpool3(out)
        out=self.conv3(out)
        out=self.avgpool(out)
        out=out.view(out.size(0),-1)
        out=self.linear1(out)
        out=self.linear1(out)
        out=self.linear2(out)
        return out

def test():
    net=schirrmeister(2)
    x=torch.randn(1,1,20,401)
    y=net(x)
    print(y.size())
class schirrmeisterComp(nn.Module):
    def __init__(self,outputSize):
        self.outputSize=outputSize
        super(schirrmeisterComp,self).__init__()
        self.preLayers=nn.Sequential(
            nn.Conv2d(1,25,kernel_size=[1,10],padding=0),
            nn.BatchNorm2d(25),
            nn.ELU(True),
        )
        self.spatial=nn.Sequential(
            nn.Conv2d(25,25,kernel_size=[20,1],padding=0),
            nn.BatchNorm2d(25),
            nn.ELU(True),
        )
        self.maxpool1=nn.AvgPool2d([1,3],stride=[1,3],padding=0)
        self.conv1=nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(25,50,kernel_size=[1,10],padding=0),
            nn.BatchNorm2d(50),
            nn.ELU(True),
        )
        self.maxpool2=nn.AvgPool2d([1,3],stride=[1,3],padding=0)
        self.conv2=nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(50,100,kernel_size=[1,10],padding=0),
            nn.BatchNorm2d(100),
            nn.ELU(True),
        )
        self.maxpool3=nn.AvgPool2d([1,3],stride=[1,3],padding=0)
        self.conv3=nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(100,200,kernel_size=[1,10],padding=0),
            nn.ELU(True),
        )
        self.avgpool=nn.AvgPool2d(1,padding=0)
        self.linear=nn.Linear(200,self.outputSize)
    def forward(self,x):
        out=self.preLayers(x)
        out=self.spatial(out)
        out=self.maxpool1(out)
        out=self.conv1(out)
        out=self.maxpool2(out)
        out=self.conv2(out)
        out=self.maxpool3(out)
        out=self.conv3(out)
        out=self.avgpool(out)
        out=out.view(out.size(0),-1)
        out=self.linear(out)
        out=out.view(out.size(0),3,-1)
        return out