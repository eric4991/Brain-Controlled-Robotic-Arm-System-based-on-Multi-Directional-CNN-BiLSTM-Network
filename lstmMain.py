import numpy as np
import scipy.io
import torch
import torch.utils.data
import torch.nn as nn
import argparse
import torch.backends.cudnn as cudnn
from sklearn.metrics import matthews_corrcoef
import matplotlib.pyplot as plt
import os

inputSize=200
numLayers=2
dropout=0.5
BIDIRECTIONAL=True
NUM_DIRS=2 if BIDIRECTIONAL else 1
saveEvery=1

class lstmV1(nn.Module):
    def __init__(self,num_tags,hiddenSize):
        super(lstmV1,self).__init__()
        self.hiddenSize=hiddenSize
        self.lstm=nn.LSTM(
            input_size=inputSize,
            hidden_size=self.hiddenSize//NUM_DIRS,
            num_layers=numLayers,
            bias=False,
            dropout=dropout,
            bidirectional=BIDIRECTIONAL
        )
        self.fc=nn.Linear(self.hiddenSize,num_tags)

    def initHidden(self):
        h = torch.zeros(numLayers * NUM_DIRS, 3, self.hiddenSize // NUM_DIRS).cuda()
        c = torch.zeros(numLayers * NUM_DIRS, 3, self.hiddenSize // NUM_DIRS).cuda()
        return (h,c)
    def forward(self,x):
        self.hidden=self.initHidden()
        out,_=self.lstm(x,self.hidden)
        out=self.fc(out)
        return out

class lstmV2(nn.Module):
    def __init__(self, num_tags, hiddenSize):
        super(lstmV2, self).__init__()
        self.hiddenSize = hiddenSize
        self.lstm = nn.LSTM(
            input_size=inputSize,
            hidden_size=self.hiddenSize // NUM_DIRS,
            num_layers=numLayers,
            bias=False,
            dropout=dropout,
            bidirectional=BIDIRECTIONAL
        )
        self.fc = nn.Sequential(
            nn.Linear(self.hiddenSize, num_tags),
            nn.ReLU(True))
        self.fc1 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(num_tags, num_tags)
        )

    def initHidden(self):
        h = torch.zeros(numLayers * NUM_DIRS, 3, self.hiddenSize // NUM_DIRS).cuda()
        c = torch.zeros(numLayers * NUM_DIRS, 3, self.hiddenSize // NUM_DIRS).cuda()
        return (h, c)

    def forward(self, x):
        self.hidden = self.initHidden()
        out, _ = self.lstm(x, self.hidden)
        out = self.fc(out)
        out = self.fc1(out)
        return out

class lstmV3(nn.Module):
    def __init__(self,num_tags,hiddenSize):
        super(lstmV3,self).__init__()
        self.hiddenSize=hiddenSize
        self.lstm=nn.LSTM(
            input_size=inputSize,
            hidden_size=self.hiddenSize//NUM_DIRS,
            num_layers=numLayers,
            bias=False,
            dropout=dropout,
            bidirectional=BIDIRECTIONAL
        )
        self.fc=nn.Sequential(
            nn.Linear(self.hiddenSize,450),
            nn.ReLU(True))
        self.fc1=nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(450,450))
        self.fc2=nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(450,num_tags)
        )


    def initHidden(self):
        h = torch.zeros(numLayers * NUM_DIRS, 3, self.hiddenSize // NUM_DIRS).cuda()
        c = torch.zeros(numLayers * NUM_DIRS, 3, self.hiddenSize // NUM_DIRS).cuda()
        return (h,c)
    def forward(self,x):
        self.hidden=self.initHidden()
        out,_=self.lstm(x,self.hidden)
        out=self.fc(out)
        out=self.fc1(out)
        out=self.fc2(out)
        return out

def test():
    x=torch.randn(1,3,200)
    net=lstmV2(301,600)
    y=net(x)
    print(y.size())
#test()