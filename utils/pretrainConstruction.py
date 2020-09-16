import numpy as np
def pretrainGTConstruction(inputTrainY,pretrainMode):
    pretrainY=np.zeros((inputTrainY.size(0),inputTrainY.size(1)))
    if pretrainMode==0:
        ## +(0) & -(1)
        for idx, row in enumerate(inputTrainY[:,:,-1]):
            for xyz in range(3):
                if row[xyz]>=0:
                    pretrainY[idx,xyz]=0
                else:
                    pretrainY[idx,xyz]=1
    elif pretrainMode==1:
        numList=[]
        for idx,row in enumerate(inputTrainY[:,:,-1]):
            for xyz in range(3):
                if row[xyz] not in numList:
                    numList=np.append(numList,row[xyz])
                    pretrainY[idx,xyz]=len(numList)-1
                else:
                    # find the indices from the numList and save into pretrainY
                    for indices,data in enumerate(numList):
                        if data==row[xyz]:
                            pretrainY[idx,xyz]=indices
    else:
        print('pretrainMode is not defined')
    return pretrainY