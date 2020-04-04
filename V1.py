import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchsummary import summary

class V1(nn.Module):
    def __init__(self,outputSize):
        self.outputSize=outputSize
        super(V1,self).__init__()
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
        self.maxpool1=nn.AvgPool2d([1,3],padding=0)
        self.conv1=nn.Sequential(
            nn.Dropout2d(p=0.5),
            nn.Conv2d(25,50,kernel_size=[1,5],padding=0),
            nn.BatchNorm2d(50),
            nn.ReLU(True),
            nn.Conv2d(50,50,kernel_size=[1,5],padding=0),
            nn.BatchNorm2d(50),
            nn.ReLU(True)
        )
        self.maxpool2=nn.AvgPool2d([1,3],padding=0)
        self.conv2=nn.Sequential(
            nn.Dropout2d(p=0.5),
            nn.Conv2d(50,100,kernel_size=[1,5],padding=0),
            nn.BatchNorm2d(100),
            nn.ReLU(True),
            nn.Conv2d(100,100,kernel_size=[1,5],padding=0),
            nn.BatchNorm2d(100),
            nn.ReLU(True),
        )
        self.maxpool3=nn.MaxPool2d([1,3],padding=0)
        self.conv3=nn.Sequential(
            nn.Dropout2d(p=0.5),
            nn.Conv2d(100,200,kernel_size=[1,5],padding=0),
            nn.BatchNorm2d(200),
            nn.ReLU(True),
            nn.Conv2d(200,200,kernel_size=[1,5],padding=0),
            nn.BatchNorm2d(200),
            nn.ReLU(True),
        )
        self.avgpool=nn.AvgPool2d([1,2],padding=0)
        self.dense=nn.Sequential(
            nn.Linear(200,200),
            nn.ReLU(True))
        self.linear=nn.Linear(200,outputSize)
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
        out=self.dense(out)
        out=self.linear(out)
        return out

class V2(nn.Module):
    def __init__(self,outputSize):
        self.outputSize=outputSize
        super(V2,self).__init__()
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
        self.avgpool1=nn.AvgPool2d([1,3],stride=[1,3],padding=0)
        self.conv1=nn.Sequential(
            nn.Dropout2d(p=0.5),
            nn.Conv2d(25,50,kernel_size=[1,10],padding=0),
            nn.BatchNorm2d(50),
            nn.ELU(True),
            nn.Conv2d(50,50,kernel_size=[1,10],padding=0),
            nn.BatchNorm2d(50),
            nn.ELU(True),
        )
        self.avgpool2=nn.AvgPool2d([1,3],stride=[1,3],padding=0)
        self.conv2=nn.Sequential(
            nn.Dropout2d(p=0.5),
            nn.Conv2d(50,100,kernel_size=[1,10],padding=0),
            nn.BatchNorm2d(100),
            nn.ELU(True),
            nn.Conv2d(100,100,kernel_size=[1,10],padding=0),
            nn.BatchNorm2d(100),
            nn.ELU(True),
        )
        self.avgpool3=nn.AvgPool2d([1,3],stride=[1,3],padding=0)
        self.fc=nn.Sequential(
            nn.Linear(600,600),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(600,300),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(300,300)
        )
    def forward(self, x):
        out=self.preLayers(x)
        out=self.spatial(out)
        out=self.avgpool1(out)
        out=self.conv1(out)
        out=self.avgpool2(out)
        out=self.conv2(out)
        out=self.avgpool3(out)
        out=out.view(out.size(0),-1)
        out=self.fc(out)
        return out
def test():
    net=V2(2)
    x=torch.randn(1,1,20,401)
    y=net(x)
    print(y.size())
#test()