import os
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import scipy.io
from functools import partial
from operator import is_not

from utils import *
from PretrainModel import *
from MainModel import *
from Model import *

pretrainMode=0

subjid = 3
sessionId=1
classId = 1
dirFileId=1
hzId=7

trainingEpoch = 300
totalTraining=1
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
device='cuda: 1'

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


#### get the last
# getting the components of trainset
def train(epoch):
    print('\nEpoch: %d' %epoch)
    net.train()
    trainLoss=0
    correct=0
    total=0
    for batchIdx, (inputs,targets) in enumerate(trainloader):
        inputs=inputs[:,np.newaxis,:,:]
        inputs,targets=inputs.to(device,dtype=torch.float),targets.to(device,dtype=torch.float)
        optimizer.zero_grad()
        outputs=net(inputs)
        loss=criterion(outputs,targets)
        loss.backward()
        optimizer.step()
        trainLoss+=loss.item()
def test(epoch):
    global trainingEpoch,lowestLoss
    net.eval()
    testLoss=0
    correct=0
    total=0
    with torch.no_grad():
        for batchIdx,(inputs,targets) in enumerate(testloader):
            inputs=inputs[:,np.newaxis,:,:]
            inputs,targets=inputs.to(device,dtype=torch.float),targets.to(device,dtype=torch.float)
            outputs=net(inputs)
            loss=criterion(outputs,targets)
            testLoss+=loss.item()
    print(testLoss)
    if lowestLoss>testLoss and epoch>minEpoch:
        print('Svaing...')
        state={
            'net':net.state_dict(),
            'loss':testLoss,
            'epoch':epoch,
        }
        if not os.path.isdir('conventionalModel'):
            os.mkdir('conventionalModel')
        torch.save(state,'./conventionalModel/ckpt_class'+str(classId)+'_subj'+str(subjid)+'_'+hzFile[hzId]+'.t7')
        lowestLoss=testLoss
    return testLoss

def visualize():
    assert os.path.isdir('conventionalModel'),'Error: no checkpoint directory found'
    checkpoint=torch.load('./conventionalModel/ckpt_class'+str(classId)+'_subj'+str(subjid)+'_'+hzFile[hzId]+'.t7')
    net.load_state_dict(checkpoint['net'])
    net.eval()
    with torch.no_grad():
        for batchIdx,(inputs,targets) in enumerate(visualoader):
            inputs=inputs[:,np.newaxis,:,:]
            inputs,targets=inputs.to(device,dtype=torch.float),targets.to(device,dtype=torch.float)
            outputs=net(inputs)
            visList=outputs.view(outputs.size(1),-1)    # 3 9600
            xPred=outputs[:,0,:]
            yPred=outputs[:,1,:]
            zPred=outputs[:,2,:]
            xTar = targets[:, 0, :]
            yTar = targets[:, 1, :]
            zTar = targets[:, 2, :]
            xR=[];yR=[];zR=[];
            for tIdx,xtar in enumerate(xTar):
                xR=np.append(xR,np.corrcoef(xTar[tIdx,:].data.cpu().numpy(),xPred[tIdx,:].data.cpu().numpy())[0,1])
                yR=np.append(yR,np.corrcoef(yTar[tIdx,:].data.cpu().numpy(),yPred[tIdx,:].data.cpu().numpy())[0,1])
                zR=np.append(zR,np.corrcoef(zTar[tIdx,:].data.cpu().numpy(),zPred[tIdx,:].data.cpu().numpy())[0,1])
            steps = np.asarray(range(visList.size(1)))
            xPred=visList[0,:]
            yPred=visList[1,:]
            zPred=visList[2,:]
            plt.figure(1)
            plt.plot(steps, xTar.data.cpu().numpy().flatten(), 'r-')
            plt.plot(steps, xPred.data.cpu().numpy().flatten(), 'b-')
            plt.figure(2)
            plt.plot(steps, yTar.data.cpu().numpy().flatten(), 'r-')
            plt.plot(steps, yPred.data.cpu().numpy().flatten(), 'b-')
            plt.figure(3)
            plt.plot(steps, zTar.data.cpu().numpy().flatten(), 'r-')
            plt.plot(steps, zPred.data.cpu().numpy().flatten(), 'b-')
            print('Average R-Value of X for each Trials: ',np.average(xR))
            print('Average R-Value of Y for each Trials: ',np.average(yR))
            print('Average R-Value of Z for each Trials: ',np.average(zR))
            scipy.io.savemat('./ForCC/sub'+str(subjid)+'session'+str(sessionId)+".mat",mdict={'groundTruth':targets.data.cpu().numpy(),'predicted':outputs.data.cpu().numpy()})
    plt.show()
    return outputs,targets

net='hi'
TestLoss=[]
for trainingIdx in range(totalTraining):
    del net
    print(str(trainingIdx)+'th training')
    print('==> Building model...')
    tmpTestLoss=[]
    #net=eegNet(903).to(device)
    net=schirrmeisterComp(903).to(device)
    if 'cuda' or 'cuda: 0' or 'cuda: 1'==device:
        cudnn.benchmark=True
    criterion=nn.MSELoss()
    optimizer=torch.optim.Adam(net.parameters(),lr=LR,weight_decay=1e-4)
    for epoch in range(trainingEpoch):
        train(epoch)
        if epoch<minEpoch:
            test(epoch)
        else:
            tmpTestLoss=np.append(tmpTestLoss,test(epoch))
    tmpTestLoss=filter(partial(is_not),tmpTestLoss)
    TestLoss=np.append(TestLoss,np.amin(tmpTestLoss))
    print('current lowest loss is: ',lowestLoss)

predVelo=visualize()
print('lowest loss is: ',lowestLoss)