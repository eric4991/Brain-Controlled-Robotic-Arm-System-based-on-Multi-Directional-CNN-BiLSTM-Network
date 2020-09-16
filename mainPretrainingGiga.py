
'''
Main function of overall pretraining and lstm after pretraining
'''

import numpy as np
import scipy.io
import torch
import torch.utils.data
import torch.nn as nn
import argparse
import torch.backends.cudnn as cudnn
import torch.optim as optim
from sklearn.metrics import matthews_corrcoef
import os

from utils import *
from PretrainModel import *

## if pretrainMode is 0, only for +(1) & -(2)
# if pretrainMode is 1, final value of each regression
pretrainMode=0

########################################################################################################################################################
# Hyper train&test parameters
#######################################################################################################################################################
subjid =10
sessionId=2
classId = 1
dirFileId=0
hzId=7
device='cuda'

trainingEpoch = 500
totalTraining=50

lowestLoss=100
lowestCC=0
start_epoch = 0
LR=1e-3
batchSize=32

hiddenSize=600
minEpoch=15
best_accX=0
best_accY=0
best_accZ=0

dirFileR=['MI','realMove']
cFile=['JH_LRFB','JH_LRFBUD']
dirFile=['MI','ME','MI_less','ME_less']
hzFile=['no','[0.1 40]','[0.1 1]','[4 7]','[8 15]','[8 30]','[16 30]','[4 40]']
classes = [['Left','Right','Forward','Backward'],['Left','Right','Forward','Backward','Up','Down']]
classNum = len(classes[classId])

traindd='./'+cFile[classId]+'/'+dirFile[dirFileId]+'/train/'+hzFile[hzId]+'/session'+str(sessionId+1)+'_sub'+str(subjid+1)+'_reaching_'+dirFileR[dirFileId]
testdd='./'+cFile[classId]+'/'+dirFile[dirFileId]+'/test/'+hzFile[hzId]+'/session'+str(sessionId+1)+'_sub'+str(subjid+1)+'_reaching_'+dirFileR[dirFileId]

# load data
## the data should contain 3 domains X, Y with label, and Y with velocity
trainData=scipy.io.loadmat(traindd)
testData=scipy.io.loadmat(testdd)
trainX = np.array(trainData['trainX'])          # 401 20 128
trainY = np.array(trainData['trainY'])          # 300 128
testX = np.array(testData['testX'])             # 301 64 32
testY = np.array(testData['testY'])             # 300 32
#testLabel=np.array(testData['testLabel'])       # 1 32

parser = argparse.ArgumentParser(description='EEG Reaching')
parser.add_argument('--lr',default=0.1, type=float,help='learning rate')
parser.add_argument('--resume','-r',action='store_true',help='resume from checkpoint')
args = parser.parse_args()

## trial in the front
trainX = np.transpose(trainX,(2,1,0))       # 128 20 401
testX = np.transpose(testX,(2,1,0))         # 32 20 401
trainY = np.transpose(trainY,(2,1,0))       # 128 3 301
testY = np.transpose(testY,(2,1,0))         # 32 3 301
#testLabel=np.transpose(testLabel,(1,0))     # 32 1

visualizeX=testX[:,:,:]
visualizeY=testY[:,:,:]

# nd array to torch tensor
trainX=torch.from_numpy(trainX)
testX=torch.from_numpy(testX)
trainY=torch.from_numpy(trainY)
testY=torch.from_numpy(testY)
visualizeX=torch.from_numpy(visualizeX)
visualizeY=torch.from_numpy(visualizeY)

#### pretrain ground truth construction
pretrainGT=pretrainGTConstruction(trainY,pretrainMode)
pretestGT=pretrainGTConstruction(testY,pretrainMode)

pretrainGTX,pretrainGTY,pretrainGTZ=pretrainGT[:,0],pretrainGT[:,1],pretrainGT[:,2]
pretestGTX,pretestGTY,pretestGTZ=pretestGT[:,0],pretestGT[:,1],pretestGT[:,2]
pretrainGTX=torch.from_numpy(pretrainGTX)
pretrainGTY=torch.from_numpy(pretrainGTY)
pretrainGTZ=torch.from_numpy(pretrainGTZ)
pretestGTX=torch.from_numpy(pretestGTX)
pretestGTY=torch.from_numpy(pretestGTY)
pretestGTZ=torch.from_numpy(pretestGTZ)
#######################################################################################################################################################
# encapsulate into a tensordataset
train=torch.utils.data.TensorDataset(trainX,trainY)
test=torch.utils.data.TensorDataset(testX,testY)

pretrainX=torch.utils.data.TensorDataset(trainX,pretrainGTX)
pretrainY=torch.utils.data.TensorDataset(trainX,pretrainGTY)
pretrainZ=torch.utils.data.TensorDataset(trainX,pretrainGTZ)
pretestX=torch.utils.data.TensorDataset(testX,pretestGTX)
pretestY=torch.utils.data.TensorDataset(testX,pretestGTY)
pretestZ=torch.utils.data.TensorDataset(testX,pretestGTZ)
visual=torch.utils.data.TensorDataset(visualizeX,visualizeY)

trainloader=torch.utils.data.DataLoader(train,batch_size=batchSize,shuffle=True)
testloader=torch.utils.data.DataLoader(test,batch_size=batchSize,shuffle=True)
visualoader=torch.utils.data.DataLoader(visual,batch_size=batchSize,shuffle=True)
pretrainloaderX=torch.utils.data.DataLoader(pretrainX,batch_size=batchSize,shuffle=True)
pretrainloaderY=torch.utils.data.DataLoader(pretrainY,batch_size=batchSize,shuffle=True)
pretrainloaderZ=torch.utils.data.DataLoader(pretrainZ,batch_size=batchSize,shuffle=True)
pretestloaderX=torch.utils.data.DataLoader(pretestX,batch_size=batchSize,shuffle=True)
pretestloaderY=torch.utils.data.DataLoader(pretestY,batch_size=batchSize,shuffle=True)
pretestloaderZ=torch.utils.data.DataLoader(pretestZ,batch_size=batchSize,shuffle=True)

def preTrainX(epoch,loader):
    print('\nEpoch: %d' %epoch)
    net.train()
    trainLoss=0
    correct=0
    total=0
    for batchIdx,(inputs,targets) in enumerate(loader):
        inputs=inputs[:,np.newaxis,:,:]
        inputs,targets=inputs.to(device,dtype=torch.float),targets.to(device,dtype=torch.long)
        optimizer.zero_grad()
        outputs=net(inputs)
        loss=criterion(outputs,targets)
        loss.backward()
        optimizer.step()

        trainLoss+=loss.item()
        _,predicted=outputs.max(1)
        total+=targets.size(0)
        correct+=predicted.eq(targets).sum().item()
def preTestX(epoch,loader,best,ver):
    global trainingEpoch
    net.eval()
    testLoss=0
    correct=0
    total=0
    with torch.no_grad():
        for batchIdx,(inputs,targets) in enumerate(loader):
            inputs=inputs[:,np.newaxis,:,:]
            inputs,targets=inputs.to(device,dtype=torch.float),targets.to(device,dtype=torch.long)
            outputs=net(inputs)
            loss=criterion(outputs,targets)
            testLoss+=loss.item()
            _,predicted=outputs.max(1)
            total+=targets.size(0)
            correct+=predicted.eq(targets).sum().item()
    acc=100*correct/total
    print(acc)
    if acc>best and epoch>minEpoch:
        print('Saving...')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('ckptPretrain'):
            os.mkdir('ckptPretrain')
        torch.save(state,'./ckptPretrain/ckpt'+ver+'_class'+str(classId)+'_subj'+str(subjid)+'_'+hzFile[hzId]+'.t7')
        best=acc
    return best

net='hi'

for trainingIdx in range(totalTraining):
    del net
    print(str(trainingIdx)+'th training')
    print('==> Building Model...')
    net=schirrmeister(2)
    net=net.to(device)
    if 'cuda'==device:
        cudnn.benchmark=True
    criterion=nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9, weight_decay=5e-4)
    for epoch in range(start_epoch,start_epoch+trainingEpoch):
        preTrainX(epoch,pretrainloaderX)
        best_accX=preTestX(epoch,pretestloaderX,best_accX,'X')
    print('current best acc for X axis is: ',best_accX)

for trainingIdx in range(totalTraining):
    del net
    print(str(trainingIdx)+'th training')
    print('==> Building Model...')
    net=schirrmeister(2)
    net=net.to(device)
    cudnn.benchmark=True
    criterion=nn.CrossEntropyLoss()
    optimizer=optim.SGD(net.parameters(),lr=0.0001, momentum=0.9, weight_decay=5e-4)
    for epoch in range(start_epoch,start_epoch+trainingEpoch):
        preTrainX(epoch,pretrainloaderY)
        best_accY=preTestX(epoch,pretestloaderY,best_accY,'Y')
    print('current best acc for Y axis is: ',best_accY)

for trainingIdx in range(totalTraining):
    del net
    print(str(trainingIdx)+'th training')
    print('==> Building Model...')
    net=schirrmeister(2)
    net=net.to(device)
    cudnn.benchmark=True
    criterion=nn.CrossEntropyLoss()
    optimizer=optim.SGD(net.parameters(),lr=0.0001, momentum=0.9, weight_decay=5e-4)
    for epoch in range(start_epoch,start_epoch+trainingEpoch):
        preTrainX(epoch,pretrainloaderZ)
        best_accZ=preTestX(epoch,pretestloaderZ,best_accZ,'Z')
    print('current best acc for Z axis is: ', best_accZ)
print('best acc for X is : ',best_accX)
print('best acc for Y is : ',best_accY)
print('best acc for Z is : ',best_accZ)