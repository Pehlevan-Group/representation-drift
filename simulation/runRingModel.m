function runRingModel(runType, Np, alp, noiseStd, learnRate, repeats,BatchSize)
% this program run the simulation to estimate the diffusion constants of a
% 1D ring model
% runType   a string alpha, Np, learnRate, sigma
% Np        number of place cells, default 1
% alp       default 0
% sigma     noise standard deviation, default 0
% learnRate leanring rate, default 0.01
% This simulation generate data for Fig.4C and Fig. S2

%%First, alpha dependent effect
if strcmp(runType,'alpha')

alps = 0:0.05:0.95;
% alps = 0.9:0.3:1;


% repeats = 10;
allDs = nan(Np,repeats,length(alps));
meanAmp = nan(length(alps),repeats);
allMaxMSD = cell(length(alps),repeats);
allRFwidth = cell(length(alps),repeats);
actiFrac = nan(length(alps),repeats);
allDphiCorr = cell(length(alps),repeats);


for i = 1:length(alps)
    for j = 1:repeats
%         [Ds, ~, pkAmp, ~]=ringModelCluster(Np,noiseStd,alps(i),learnRate,BatchSize);
        [Ds,~, pkAmp, RFwidth,maxMSD,corr_dphi]=ringModelCluster(Np,noiseStd,alps(i),learnRate,BatchSize);
            allDs(:,j,i) = Ds;
            meanAmp(i,j) = nanmean(pkAmp(:));
            actiFrac(i,j) = mean(~isnan(pkAmp(:)));
            allRFwidth{i,j} = RFwidth;
            allMaxMSD{i,j} = maxMSD;  % maximum msds
            allDphiCorr{i,j} = corr_dphi;
    end
end

elseif strcmp(runType,'learnRate')

% learning rate dependence
% Np = 1;      % start from single neurons
% noiseStd = 0;
lrs = 10.^(-3:0.25:-1);

% repeats = 10;
allDs = nan(Np,repeats,length(lrs));
meanAmp = nan(length(lrs),repeats);
allMaxMSD = cell(length(lrs),repeats);
allRFwidth = cell(length(lrs),repeats);
actiFrac = nan(length(lrs),repeats);   % store the avearge fraction of active
allDphiCorr = cell(length(lrs),repeats);


for i = 1:length(lrs)
    for j = 1:repeats
%         [Ds, ~, pkAmp, ~]=ringModelCluster(Np,noiseStd,alp,lrs(i),BatchSize);
        [Ds,~, pkAmp, RFwidth,maxMSD,corr_dphi]=ringModelCluster(Np,noiseStd,alp,lrs(i),BatchSize);
            allDs(:,j,i) = Ds;
            meanAmp(i,j) = nanmean(pkAmp(:));
            actiFrac(i,j) = mean(~isnan(pkAmp(:)));
            allRFwidth{i,j} = RFwidth;
            allMaxMSD{i,j} = maxMSD;  % maximum msds
            allDphiCorr{i,j} = corr_dphi;
    end
end

% number of different output neurons
elseif strcmp(runType,'Np')
    % when Np <50 , repeats,
%     Nps = 2.^(0:9);
%     Nps = 200;
    Nps = [2,10,20,50,100,200,400,600,800,1000];
%     Nps = [10,50];
    allDs = cell(length(Nps),repeats);
    allAmp = cell(length(Nps),repeats);
    allMaxMSD = cell(length(Nps),repeats);
    allRFwidth = cell(length(Nps),repeats);
    actiFrac = nan(length(Nps),repeats);   % store the avearge fraction of active
    meanAmp = nan(length(Nps),repeats);
    allDphiCorr = cell(length(Nps),repeats); %store correlation coefficient of dphi
    
    for i = 1:length(Nps)
        if Nps(i) <50
            for j = 1:repeats
                [Ds, ~, pkAmp,RFwidth,maxMSD,corr_dphi] = ringModelCluster(Nps(i),noiseStd,alp,learnRate,BatchSize);
                allDs{i,j} = Ds;
                allAmp{i,j} = pkAmp;
                meanAmp(i,j) = nanmean(pkAmp(:));
                actiFrac(i,j) = mean(~isnan(pkAmp(:)));
                allRFwidth{i,j} = RFwidth;
                allMaxMSD{i,j} = maxMSD;  % maximum msds
                allDphiCorr{i,j} = corr_dphi;
            end
        else
            [Ds, ~, pkAmp, RFwidth,maxMSD,corr_dphi]=ringModelCluster(Nps(i),noiseStd,alp,learnRate,BatchSize);
            allDs{i,1} = Ds;
%             allAmp{i,1} = pkAmp;
            meanAmp(i,1) = nanmean(pkAmp(:));
            actiFrac(i,1) = mean(~isnan(pkAmp(:)));
            allRFwidth{i,1} = RFwidth;
            allMaxMSD{i,1} = maxMSD;  % maximum msds
            allDphiCorr{i,1} = corr_dphi;
        end
    end
    
% different strength of noise ampltidue
elseif strcmp(runType,'sigma')
    sigs = 10.^(-4:0.25:-1);
    allDs = nan(Np,repeats,length(sigs));
%     allAmp = cell(length(sigs),repeats);
    meanAmp = nan(length(sigs),repeats);
    allMaxMSD = cell(length(sigs),repeats);
    allRFwidth = cell(length(sigs),repeats);
    actiFrac = nan(length(sigs),repeats);   % store the avearge fraction of active
    allDphiCorr = cell(length(sigs),repeats);

        
    for i = 1:length(sigs)
        for j = 1:repeats
            [Ds,~, pkAmp, RFwidth,maxMSD,corr_dphi]=ringModelCluster(Np,sigs(i),alp,learnRate,BatchSize);
            allDs(:,j,i) = Ds;
%             allAmp{i,j} = pkAmp;
            meanAmp(i,j) = nanmean(pkAmp(:));
            actiFrac(i,j) = mean(~isnan(pkAmp(:)));
            allRFwidth{i,j} = RFwidth;
            allMaxMSD{i,j} = maxMSD;  % maximum msds
            allDphiCorr{i,j} = corr_dphi;
        end
    end
end

% save the data
sFile = ['./data/pc1D_ring_',runType,'_alp',num2str(alp),'_Np',num2str(Np),...
    '_std',num2str(noiseStd),'_lr',num2str(learnRate),'_bs',num2str(BatchSize),'-',date,'.mat'];
save(sFile, '-v7.3')
end
