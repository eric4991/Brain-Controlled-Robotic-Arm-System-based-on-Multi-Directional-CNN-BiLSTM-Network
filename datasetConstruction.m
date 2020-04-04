clear all; close all; clc;
dd='.\Data';
% filelist={'dslim','eslee','jmlee','shchoi','yspark'};
filelist={'hsjeong'}

hzIdx=1;
modIdx=1;

mode={'MI','realMove'};
hz=[100 250 1000]
numClass=6;
trainRatio=0.8;

for file=1:length(filelist)
    file
    disp('Refinge...');
    
    load('.\modiedGroundTruth');
    re_velocity = re_velocity';
    re_velo = cell(6,1);
    for kk=1: size(re_velocity,1)
        re_velo{kk} = [re_velocity{kk,1} re_velocity{kk,2} re_velocity{kk,3}];
    end
    ival=[0 4000]
    [cnt,mrk,mnt]=eegfile_loadMatlab(['./ConvertedData/' mode{modIdx} '/' num2str(hz(hzIdx)) '/' filelist{file} '_reaching_' mode{modIdx}]);
    
    cnt=proc_filtButter(cnt,5,[4 40]);
    cnt=proc_commonAverageReference(cnt);   
    epo=cntToEpo(cnt,mrk,ival);
    %% Select channels
    epo = proc_selectChannels(epo, {'FC5','FC3','FC1','FC2','FC4','FC6',...
        'C5','C3','C1', 'Cz', 'C2', 'C4', 'C6',...
        'CP5','CP3','CP1','CPz','CP2','CP4','CP6'});
    
    epoLeft=proc_selectClasses(epo,{'Left'});
    epoRight=proc_selectClasses(epo,{'Right'});
    epoForward=proc_selectClasses(epo,{'Forward'});
    epoBackward=proc_selectClasses(epo,{'Backward'});
    
    epoUp=proc_selectClasses(epo,{'Up'});
    epoDown=proc_selectClasses(epo,{'Down'});
    %% DataShuffling
    epoLeft.x=datasample(epoLeft.x,size(epoLeft.x,3),3,'Replace',false);
    epoRight.x=datasample(epoRight.x,size(epoRight.x,3),3,'Replace',false);
    epoForward.x=datasample(epoForward.x,size(epoForward.x,3),3,'Replace',false);
    epoBackward.x=datasample(epoBackward.x,size(epoBackward.x,3),3,'Replace',false);
    epoUp.x=datasample(epoUp.x,size(epoUp.x,3),3,'Replace',false);
    epoDown.x=datasample(epoDown.x,size(epoDown.x,3),3,'Replace',false);
    %% train test split
    trainNum=round(size(epoLeft.x,3)*trainRatio);
    
    epoLeftTrain=proc_selectEpochs(epoLeft,1:trainNum);
    epoRightTrain=proc_selectEpochs(epoRight,1:trainNum);
    epoForwardTrain=proc_selectEpochs(epoForward,1:trainNum);
    epoBackwardTrain=proc_selectEpochs(epoBackward,1:trainNum);
    
    epoUpTrain=proc_selectEpochs(epoUp,1:trainNum);
    epoDownTrain=proc_selectEpochs(epoDown,1:trainNum);
    
    epoLeftTest=proc_selectEpochs(epoLeft,trainNum+1:size(epoLeft.x,3));
    epoRightTest=proc_selectEpochs(epoRight,trainNum+1:size(epoRight.x,3));
    epoForwardTest=proc_selectEpochs(epoForward,trainNum+1:size(epoForward.x,3));
    epoBackwardTest=proc_selectEpochs(epoBackward,trainNum+1:size(epoBackward.x,3));
    
    epoUpTest=proc_selectEpochs(epoUp,trainNum+1:size(epoUp.x,3));
    epoDownTest=proc_selectEpochs(epoDown,trainNum+1:size(epoDown.x,3));
        %% Data Construction for Training
    trainLR=proc_appendEpochs(epoLeftTrain, epoRightTrain); trainFB=proc_appendEpochs(epoForwardTrain, epoBackwardTrain);
    trainUD=proc_appendEpochs(epoUpTrain, epoDownTrain);
    trainLRFB=proc_appendEpochs(trainLR,trainFB);
    trainLRFBUD = proc_appendEpochs(trainLRFB,trainUD);
    
    switch(numClass)
        case 4
            %% %%%%%%%%% 4-class %%%%%%%%%%%
            trainX=trainLRFB.x;
            for ytr = 1: size(trainLRFB.y,2)
                b = find(trainLRFB.y(:,ytr)==1);
                trainY(:,:,ytr) = re_velo{b,1};
            end
        case 6
            %% %%%%%%%%% 6-class %%%%%%%%%%%
            trainX=trainLRFBUD.x;
            for ytr = 1: size(trainLRFBUD.y,2)
                b = find(trainLRFBUD.y(:,ytr)==1);
                trainY(:,:,ytr) = re_velo{b,1};
            end
    end
    %% Data Construction for Testing
    testLR=proc_appendEpochs(epoLeftTest, epoRightTest); testFB = proc_appendEpochs(epoForwardTest, epoBackwardTest);
    testUD=proc_appendEpochs(epoUpTest, epoDownTest);
    testLRFB=proc_appendEpochs(testLR,testFB);
    testLRFBUD = proc_appendEpochs(testLRFB, testUD);
    
    switch(numClass)
        %% %%%%%%%%% 4-class %%%%%%%%%%%
        case 4
            testX=testLRFB.x;
            for ytst = 1: size(testLRFB.y,2)
                b = find(testLRFB.y(:,ytst)==1);
                testY(:,:,ytst) = re_velo{b,1};
            end
            %% %%%%%%%%% 6-class %%%%%%%%%%%
        case 6
            testX=testLRFBUD.x;
            for ytst = 1: size(testLRFBUD.y,2)
                b = find(testLRFBUD.y(:,ytst)==1);
                testY(:,:,ytst) = re_velo{b,1};
            end
    end
    try
        save(['./DLData/' mode{modIdx} '\train\' filelist{file}],'trainX','trainY');
        save(['./DLData/' mode{modIdx} '\test\' filelist{file}],'testX','testY');
        clear trainX trainY testX testY;
        disp('finish saving')
    catch
        mkdir(['./DLData/' mode{modIdx} '\train']);
        mkdir(['./DLData/' mode{modIdx} '\test']);
        save(['./DLData/' mode{modIdx} '\train\' filelist{file}],'trainX','trainY');
        save(['./DLData/' mode{modIdx} '\test\' filelist{file}],'testX','testY');
        clear trainX trainY testX testY;
        disp('finish saving')
    end
end