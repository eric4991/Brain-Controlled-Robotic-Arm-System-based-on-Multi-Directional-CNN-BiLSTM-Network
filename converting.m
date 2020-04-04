clear all; close all; clc;
dd='.\Data';
% filelist={'dslim','eslee','jmlee','shchoi','yspark'};
filelist={'hsjeong'}

hzIdx=1;
modIdx=1;

mode={'MI','realMove'};
hz=[100 250 1000]

for ff=1:length(filelist)
    file=filelist{ff}
    opt=[];
    fprintf('** Processing of %s **\n', [dd '\' file '\' file '_reaching_' mode{modIdx}]);
        
    try
        hdr=eegfile_readBVheader([dd '\' file '\' file '_reaching_' mode{modIdx}]);
    catch
        fprintf('file could not found');
        continue;
    end
    Wps= [42 49]/hdr.fs*2;
    [n, Ws]= cheb2ord(Wps(1), Wps(2),3, 40);
    [filt.b, filt.a]= cheby2(n, 50, Ws);
    [cnt,mrk_orig]=eegfile_loadBV([dd '\' file '\' file '_reaching_' mode{modIdx}],...
        'filt',filt,'clab',{'not','EMG*'},'fs',hz(hzIdx));    
    cnt.title=['./ConvertedData/' mode{modIdx} '/' num2str(hz(hzIdx)) '/' file '_reaching_' mode{modIdx}];
    mrk=imageArrow(mrk_orig,1);
    mnt=getElectrodePositions(cnt.clab);
    fs_orig=mrk_orig.fs;
    var_list={'fs_orig',fs_orig,'mrk_orig',mrk_orig,'hdr',hdr};
    eegfile_saveMatlab(cnt.title,cnt,mrk,mnt,...
        'channelwise',1,...
        'format','int16',...
        'resolution',NaN);
end
disp('All EEG Data Converting was Done');
    