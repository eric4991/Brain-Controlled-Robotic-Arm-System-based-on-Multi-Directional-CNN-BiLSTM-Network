
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
from scipy import integrate
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

from collections import OrderedDict
import os
from functools import partial
from operator import is_not

from utils import *
from PretrainModel import *
from MainModel import *

## if pretrainMode is 0, only for +(1) & -(2)
# if pretrainMode is 1, final value of each regression
pretrainMode=0

########################################################################################################################################################
# Hyper train&test parameters
#######################################################################################################################################################
subjid =20
sessionId=0
classId = 1
dirFileId=0
hzId=7

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
device='cuda'

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
visualoader=torch.utils.data.DataLoader(visual,batch_size=visualizeX.size(0),shuffle=True)
pretrainloaderX=torch.utils.data.DataLoader(pretrainX,batch_size=batchSize,shuffle=True)
pretrainloaderY=torch.utils.data.DataLoader(pretrainY,batch_size=batchSize,shuffle=True)
pretrainloaderZ=torch.utils.data.DataLoader(pretrainZ,batch_size=batchSize,shuffle=True)
pretestloaderX=torch.utils.data.DataLoader(pretestX,batch_size=batchSize,shuffle=True)
pretestloaderY=torch.utils.data.DataLoader(pretestY,batch_size=batchSize,shuffle=True)
pretestloaderZ=torch.utils.data.DataLoader(pretestZ,batch_size=batchSize,shuffle=True)

###################################################################################################################################################
# load pretrained model
modelX=schirrmeister(2)
modelY=schirrmeister(2)
modelZ=schirrmeister(2)

checkpointX=torch.load('./ckptPretrain/ckptX'+'_class'+str(classId)+'_subj'+str(subjid)+'_'+hzFile[hzId]+'.t7')
checkpointY=torch.load('./ckptPretrain/ckptY'+'_class'+str(classId)+'_subj'+str(subjid)+'_'+hzFile[hzId]+'.t7')
checkpointZ=torch.load('./ckptPretrain/ckptZ'+'_class'+str(classId)+'_subj'+str(subjid)+'_'+hzFile[hzId]+'.t7')

modelX.load_state_dict(checkpointX['net'])
modelY.load_state_dict(checkpointY['net'])
modelZ.load_state_dict(checkpointZ['net'])

modelX=nn.Sequential(*list(modelX.children())[:-1]).cuda()
modelY=nn.Sequential(*list(modelY.children())[:-1]).cuda()
modelZ=nn.Sequential(*list(modelZ.children())[:-1]).cuda()

#### get the last
# getting the components of trainset
def train(epoch):
    global tmpLoss
    print('\nEpoch: %d' %epoch)
    modelX.eval()
    modelY.eval()
    modelZ.eval()
    rnn.train()
    trainLoss=0
    for batch_idx,(inputs,targets) in enumerate(trainloader):
        inputs=inputs[:,np.newaxis,:,:]
        inputs,targets=inputs.to(device,dtype=torch.float),targets.to(device,dtype=torch.float)
        aftPretrainedX=modelX(inputs)
        aftPretrainedY=modelY(inputs)
        aftPretrainedZ=modelZ(inputs)
        trainedInputs=torch.squeeze(torch.cat([aftPretrainedX,aftPretrainedY,aftPretrainedZ],2))
        trainedInputs=trainedInputs.permute(0,2,1)
        optimizer.zero_grad()
        outputs=rnn(trainedInputs)
        loss=lossFunc(outputs,targets)
        loss.backward()
        optimizer.step()
        trainLoss+=loss.item()
    rmseLoss=np.sqrt(trainLoss)
    tmpLoss=np.append(tmpLoss,rmseLoss)
def test(epoch):
    global lowestLoss,trainingEpoch
    modelX.eval()
    modelY.eval()
    modelZ.eval()
    rnn.eval()
    testLoss=0
    with torch.no_grad():
        for batchIdx, (inputs,targets) in enumerate(testloader):
            inputs=inputs[:,np.newaxis,:,:]
            inputs,targets=inputs.to(device,dtype=torch.float),targets.to(device,dtype=torch.float)
            aftPretestX=modelX(inputs)
            aftPretestY=modelY(inputs)
            aftPretestZ=modelZ(inputs)
            trainedInputsTest=torch.squeeze(torch.cat([aftPretestX,aftPretestY,aftPretestZ],2))
            trainedInputsTest=trainedInputsTest.permute(0,2,1)
            outputs=rnn(trainedInputsTest)
            loss=lossFunc(outputs,targets)
            testLoss+=loss.item()
    print(testLoss)
    if lowestLoss>testLoss and epoch>minEpoch:
        print('Saving...')
        state={
            'net':rnn.state_dict(),
            'loss':testLoss,
            'epoch':epoch,
        }
        if not os.path.isdir('ckptMain'):
            os.mkdir('ckptMain')
        torch.save(state,'./ckptMain/ckpt_class'+str(classId)+'_subj'+str(subjid)+'_'+hzFile[hzId]+'.t7')
        lowestLoss=testLoss
        return testLoss
def visualize():
    assert os.path.isdir('ckptMain'),'Error: no checkpoint directory found'
    checkpoint=torch.load('./ckptMain/ckpt_class'+str(classId)+'_subj'+str(subjid)+'_'+hzFile[hzId]+'.t7')
    rnn.load_state_dict(checkpoint['net'])
    modelX.eval()
    modelY.eval()
    modelZ.eval()
    with torch.no_grad():
        for batchIdx,(inputs,targets) in enumerate(visualoader):
            inputs=inputs[:,np.newaxis,:,:]
            inputs,targets=inputs.to(device,dtype=torch.float),targets.to(device,dtype=torch.float)
            aftVisX=modelX(inputs)
            aftVisY=modelY(inputs)
            aftVisZ=modelZ(inputs)
            trainedVis=torch.squeeze(torch.cat([aftVisX,aftVisY,aftVisZ],2))
            trainedVis=trainedVis.permute(0,2,1)
            outputs=rnn(trainedVis)                 # 48 3 401
            visList=outputs.view(outputs.size(1),-1)    # 3 9600
            xPred=outputs[:,0,:].data.cpu().numpy()
            yPred=outputs[:,1,:].data.cpu().numpy()
            zPred=outputs[:,2,:].data.cpu().numpy()
            xTar = targets[:, 0, :].data.cpu().numpy()
            yTar = targets[:, 1, :].data.cpu().numpy()
            zTar = targets[:, 2, :].data.cpu().numpy()
            xR=[];yR=[];zR=[]
            for tIdx,xtar in enumerate(xTar):
                xR=np.append(xR,np.corrcoef(xtar,xPred[tIdx,:])[0,1])
                yR=np.append(yR,np.corrcoef(yTar[tIdx,:],yPred[tIdx,:])[0,1])
                zR=np.append(zR,np.corrcoef(zTar[tIdx,:],zPred[tIdx,:])[0,1])
            steps = np.asarray(range(visList.size(1)))
            xPred=visList[0,:]
            yPred=visList[1,:]
            zPred=visList[2,:]
            '''
            plt.figure(1)
            plt.plot(steps, xTar.data.cpu().numpy().flatten(), 'r-')
            plt.plot(steps, xPred.data.cpu().numpy().flatten(), 'b-')
            plt.figure(2)
            plt.plot(steps, yTar.data.cpu().numpy().flatten(), 'r-')
            plt.plot(steps, yPred.data.cpu().numpy().flatten(), 'b-')
            plt.figure(3)
            plt.plot(steps, zTar.data.cpu().numpy().flatten(), 'r-')
            plt.plot(steps, zPred.data.cpu().numpy().flatten(), 'b-')
            '''
            print('Average R-Value of X for each Trials: ',np.average(xR))
            print('Average R-Value of Y for each Trials: ',np.average(yR))
            print('Average R-Value of Z for each Trials: ',np.average(zR))
            scipy.io.savemat('./ForCC/sub'+str(subjid)+'session'+str(sessionId)+".mat",mdict={'groundTruth':targets.data.cpu().numpy(),'predicted':outputs.data.cpu().numpy()})
    plt.show()
    return outputs,targets
    #return targets.data.cpu().numpy()


def posVisualization(predVel):
    predVel=predVel.data.cpu().numpy()
    steps=np.asarray(range(len(predVel[0][0])))


    ax=plt.figure().gca(projection='3d')
    gtArray=scipy.io.loadmat('./gtArray.mat')
    posArray=np.array(gtArray['gtArray'])

    for cIdx in range(len(posArray)):
        ax.plot(posArray[cIdx][0],posArray[cIdx][1],posArray[cIdx][2],'blue')

    for trialIdx in range(len(predVel)):
        if trialIdx==0:
            tarIntegX = integrate.cumtrapz(predVel[trialIdx][0], steps)
            tarIntegY = integrate.cumtrapz(predVel[trialIdx][1], steps)
            tarIntegZ = integrate.cumtrapz(predVel[trialIdx][2], steps)

            tarIntegX=tarIntegX[np.newaxis,:]
            tarIntegY=tarIntegY[np.newaxis,:]
            tarIntegZ=tarIntegZ[np.newaxis,:]
        else:
            tmpX=integrate.cumtrapz(predVel[trialIdx][0],steps)
            tmpY=integrate.cumtrapz(predVel[trialIdx][1],steps)
            tmpZ=integrate.cumtrapz(predVel[trialIdx][2],steps)
            tmpX=tmpX[np.newaxis,:]
            tmpY=tmpY[np.newaxis,:]
            tmpZ=tmpZ[np.newaxis,:]
            tarIntegX=np.append(tarIntegX,tmpX,axis=0)
            tarIntegY=np.append(tarIntegY,tmpY,axis=0)
            tarIntegZ=np.append(tarIntegZ,tmpZ,axis=0)
        ax.plot(tarIntegX[trialIdx,:],tarIntegY[trialIdx,:],tarIntegZ[trialIdx,:],'red')

    plt.show()

rnn='hi'
TestLoss=[]
visTotalLoss=[]
for trainingIdx in range(totalTraining):
    del rnn
    print(str(trainingIdx)+'th training')
    print('==> Building model...')
    tmpTestLoss=[]
    tmpLoss=[]

    rnn=lstmV2(301,600)
    rnn=rnn.to(device)
    if 'cuda'==device:
        cudnn.benchmark=True
    lossFunc=nn.MSELoss()
    optimizer=torch.optim.Adam(rnn.parameters(),lr=LR,weight_decay=1e-4)
    for epoch in range(trainingEpoch):
        train(epoch)
        if epoch<minEpoch:
            test(epoch)
        else:
            tmpTestLoss=np.append(tmpTestLoss,test(epoch))
    tmpTestLoss=filter(partial(is_not),tmpTestLoss)
    TestLoss=np.append(TestLoss,np.amin(tmpTestLoss))
    avgTmp=np.average(tmpLoss)
    if trainingIdx==0:
        avg=0
    if avgTmp>avg:
        avg=avgTmp
        visTotalLoss=tmpLoss
    print('current loswest loss is : ', lowestLoss)

#predVelo=visualize()
print('lowest loss is: ',lowestLoss)
scipy.io.savemat('./'+str(subjid)+'.mat',{'visTrainLoss':np.array(visTotalLoss)})

rnn=lstmV2(301,600)
rnn=rnn.to(device)

predVelo,target=visualize()
target=target.data.cpu().numpy()
predVelo=predVelo.data.cpu().numpy()
scipy.io.savemat('./Visualization/Pred_sub'+str(subjid)+'_hz'+str(hzId)+'.mat',{'predVelo':predVelo})
scipy.io.savemat('./Visualization/Target_sub'+str(subjid)+'_hz'+str(hzId)+'.mat',{'targetVelo':target})

'''

## Visualizing only
rnn=lstmV2(301,600)
rnn=rnn.to(device)

predVelo,target=visualize()
target=target.data.cpu().numpy()
predVelo=predVelo.data.cpu().numpy()
scipy.io.savemat('./Visualization/Pred_sub'+str(subjid)+'_hz'+str(hzId)+'.mat',{'predVelo':predVelo})
scipy.io.savemat('./Visualization/Target_sub'+str(subjid)+'_hz'+str(hzId)+'.mat',{'targetVelo':target})
'''