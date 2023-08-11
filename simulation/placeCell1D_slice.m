% using non-negative similarity matching to learn a 1-d place cells based
% on predefined grid cells patterns. Based on the paper lian et al 2020
% 1D grid fields are slices of 2D lattice of grid fields

% add input noise simulation
% last revised  8/22/2022
clear
close all

%% model parameters
param.ps =  200;        % number of positions along each dimension
param.Nlbd = 5;         % number of different scales/spacing
param.Nthe = 6;         % number of rotations
param.Nx =  3;          % offset of x-direction
param.Ny = 3;           % offset of y-direction
param.Ng = param.Nlbd*param.Nthe*param.Nx*param.Ny;   % total number of grid cells
param.Np = 200;         % number of place cells, default 20*20

param.baseLbd = 1/4;    % spacing of smallest grid RF, default 1/4
param.sf =  1.42;       % scaling factor between adjacent module

% parameters for learning 
noiseVar = 'same';      % using different noise or the same noise level for each synpase
noiseStd = 0.005;        % 0.001
learnRate = 0.02;       % default 0.05

param.W = 0.5*randn(param.Np,param.Ng);   % initialize the forward matrix
param.M = eye(param.Np);        % lateral connection if using simple nsm
param.lbd1 = 0.02;              % 0.15 for 400 place cells and 5 modes
param.lbd2 = 0.05;              % 0.05 for 400 place cells and 5 modes


param.alpha = 35;        % the threshold depends on the input dimension, 65
param.beta = 2; 
param.gy = 0.05;         % update step for y
param.gz = 0.1;          % update step for z
param.gv = 0.2;          % update step for V
param.b = zeros(param.Np,1);  % biase
param.learnRate = learnRate;  % learning rate for W and b
param.noise =  noiseStd; % stanard deivation of noise 
param.step = 20;         % store every 20 step
param.ampThd = 0.1;      % amplitude threshold, depending on the parameters

save_data_flag = false;   % store the data or not


param.sigWmax = noiseStd;% the maximum noise std of forward synapses
param.sigMmax = noiseStd;     
if strcmp(noiseVar, 'various')
%     noiseVecW = rand(param.Np,1);
    noiseVecW = 10.^(rand(param.Np,1)*2-2);
    param.noiseW = noiseVecW*ones(1,param.Ng)*param.sigWmax;   % noise amplitude is the same for each posterior 
%     noiseVecM = rand(param.Np,1);
    noiseVecM = 10.^(rand(param.Np,1)*2-2);
    param.noiseM = noiseVecM*ones(1,param.Np)*param.sigMmax; 
else
    param.noiseW =  param.sigWmax*ones(param.Np,param.Ng);    % stanard deivation of noise, same for all
    param.noiseM =  param.sigMmax*ones(param.Np,param.Np);   
end

param.ori = 1/6*pi;     % slicing orientation

param.BatchSize = 1;      % minibatch used to to learn
param.learnType = 'snsm';  % snsm, batch, online, randwalk, direction, inputNoise

gridQuality = 'slice';  % regular, weak or slice

makeAnimation = 0;    % whether make a animation or not

gridFields = slice2Dgrid(param.ps,param.Nlbd,param.Nthe,param.Nx,param.Ny,param.ori);


%% using non-negative similarity matching to learn place fields
% generate input from grid filds

total_iter = 5e3;   % total interation, default 2e3

posiGram = eye(param.ps);
gdInput = gridFields'*posiGram;  % gramin of input matrix

% Fir the initial stage, only return the updated parameters
[~, param] = place_cell_stochastic_update_snsm(gdInput,total_iter, param);


% estimate the place field of place cells
numProb = param.ps;    % number of positions used to estimate the RF

ys_prob = zeros(param.Np,1);
% ys_prob = zeros(param.Np,numProb);
ys = zeros(param.Np, numProb);
for i = 1:numProb
    states= PlaceCellhelper.nsmDynBatch(gdInput(:,i),ys_prob, param);
    ys(:,i) = states.Y;
end

%% Continuous Nosiy update
% 
total_iter = 2e4;
time_points = round(total_iter/param.step);

[output, param] = place_cell_stochastic_update_snsm(gdInput,total_iter, param, false, true, true);

%% Estimate the diffusion constant

msds = nan(floor(time_points/2),size(output.pkMas,1));
for i = 1:size(msds,1)
    
    if param.Np == 1
        t1 = abs(output.pkCenterMass(:,i+1:end)' - output.pkCenterMass(:,1:end-i)');
    else
        t1 = abs(output.pkCenterMass(:,i+1:end) - output.pkCenterMass(:,1:end-i));
    end
    
    dx = min(t1,param.ps - t1);
    msds(i,:) = nanmean(dx.^2,2);
end

% linear regression to get the diffusion constant of each neuron
Ds = PlaceCellhelper.fitLinearDiffusion(msds,param.step,'linear');


%% Diffusion constant, active time
temp = output.pkAmp > param.ampThd;
actiFrac = sum(temp,2)/size(output.pkAmp,2);


%% interval of active, silent, shift of RF

% interval length where place fields disappears
ampThd = 0.1;          % threshold to determine whether active
pkFlags = output.pkAmp > ampThd;
silentInter = [];  % store the silent interval
activeInter = [];
randActiInter = [];  % compare with random case
actInxPerm = reshape(pkFlags(randperm(size(pkFlags,1)*size(pkFlags,2))),size(pkFlags));
for i = 1:size(output.pkAmp,1)
    I =  find(pkFlags(i,:));
    temp = diff(I);
    silentInter = [silentInter,temp(temp>1)-1];
    
    Ia = find(pkFlags(i,:)==0);
    temp = diff(Ia);
    activeInter = [activeInter,temp(temp>1)-1];
    
    % for random permutation
    Ib = find(actInxPerm(i,:)==0);
    temp = diff(Ib);
    randActiInter = [randActiInter,temp(temp>1)-1];
end

% tabulate the results
randActTab = tabulate(randActiInter);
actTab = tabulate(activeInter);

%% Correlation of shift of centroids

%newPeaks = mydata;
trackL = 200;                 % length of the track
centroid_sep = 0:10:trackL;   % bins of centroid positions
deltaTau = 100;               % select time step seperation, subsampling
% numAnimals = size(pc_Centroid.allCentroids,1);
repeats = min(20,deltaTau);

shiftCentroid = cell(length(centroid_sep)-1,repeats);
aveShift = nan(length(centroid_sep)-1,repeats);

for rp = 1:repeats
    initialTime = randperm(deltaTau,1);
    % a new method to estimate the correlations
    for bin = 1:length(centroid_sep)-1
        firstCen = [];
        secondCen = [];

        for i = initialTime:deltaTau:(size(output.pkCenterMass,2)-deltaTau)

            bothActiInx = ~isnan(output.pkCenterMass(:,i+deltaTau)) & ~isnan(output.pkCenterMass(:,i));
            ds = output.pkCenterMass(bothActiInx,i+deltaTau) - output.pkCenterMass(bothActiInx,i);
            temp = output.pkCenterMass(bothActiInx,i);
            
            % remove two ends
            rmInx = temp < 0.2*trackL | temp > 0.8*trackL;
            temp(rmInx) = nan;  
            % pairwise active centroid of these two time points
            centroidDist = squareform(pdist(temp));
            [ix,iy] = find(centroidDist > centroid_sep(bin) & centroidDist <= centroid_sep(bin+1));
            firstCen = [firstCen;ds(ix)];
            secondCen = [secondCen;ds(iy)];
        end
        % merge
        shiftCentroid{bin,rp} = firstCen.*secondCen;
    %     aveShift(bin) = nanmean(firstCen.*secondCen)/nanstd(firstCen)/nanstd(secondCen);
        C = corrcoef(firstCen,secondCen);
        aveShift(bin,rp) = C(1,2);
    end
end
% avearge
aveRho = nan(length(centroid_sep)-1,repeats);

for rp = 1:repeats
    for i = 1:length(centroid_sep)-1
        aveRho(i,rp) = nanmean(shiftCentroid{i,rp});
    end
end
finalAveRho = nanmean(aveRho,2);


% Model centroid shift distribution
modelShift = [];
for rp = 1:repeats
    initialTime = randperm(deltaTau,1);
    for i = initialTime:deltaTau:(size(output.pkCenterMass,2)-deltaTau)

        bothActiInx = ~isnan(output.pkCenterMass(:,i+deltaTau)) & ~isnan(output.pkCenterMass(:,i));
        ds = output.pkCenterMass(bothActiInx,i+deltaTau) - output.pkCenterMass(bothActiInx,i);
        modelShift = [modelShift;ds];
    end
end 

%% compared with pure random walk
% the agregated data is save for further use

wkspeed = 1;   % walk speed
numNeu = 200;
repeats = 20;  % repeat
initialPosi = randi(trackL,numNeu,1);  % initial centroid poistions
totSteps = round(size(output.pkCenterMass,2)/deltaTau);
nModelSample = length(modelShift);

% centroids_randomWalk = cell(repeats,1);  % st
shiftCentroid = cell(length(centroid_sep)-1,repeats);
aveShiftM = nan(length(centroid_sep)-1,repeats);

for rp = 1:repeats
    centroids_randomWalk = nan(numNeu,totSteps);
    for i = 1:numNeu
        temp = randi(trackL,1);
        ds = randperm(length(modelShift),1);  % random generate a shift
        for j = 1:totSteps
            newTry = temp + modelShift(randperm(nModelSample,1));
            if newTry < 0
                temp = abs(newTry);
            elseif newTry > trackL
                temp = max(1,2*trackL - newTry);  % can't be negative
            else
                temp = newTry;
            end            
            
            centroids_randomWalk(i,j) = temp;
        end
    end
    
    % distance dependent correlation
    for bin = 1:length(centroid_sep)-1
        firstCen = [];
        secondCen = [];

        for i = 1:totSteps-1

            bothActiInx = ~isnan(centroids_randomWalk(:,i+1)) & ~isnan(centroids_randomWalk(:,i));
            ds = centroids_randomWalk(bothActiInx,i+1) - centroids_randomWalk(bothActiInx,i);
            temp = centroids_randomWalk(bothActiInx,i);
            
            % remove two ends
            rmInx = temp < 0.2*trackL | temp > 0.8*trackL;
            temp(rmInx) = nan;
            
            % pairwise active centroid of these two time points
            centroidDist = squareform(pdist(temp));
            [ix,iy] = find(centroidDist > centroid_sep(bin) & centroidDist <= centroid_sep(bin+1));
            firstCen = [firstCen;ds(ix)];
            secondCen = [secondCen;ds(iy)];
        end
        % merge
        shiftCentroid{bin,rp} = firstCen.*secondCen;
        C = corrcoef(firstCen,secondCen);
        aveShiftM(bin,rp) = C(1,2);
    end
end

%% SAVE THE DATA
if save_data_flag
    save_folder = fullfile(pwd,'data', filesep,'revision');
    save_data_name = fullfile(save_folder, ['pc_1D_various_noise_',date,'.mat']);
    save(save_data_name,'-v7.3')
end

%% Publication ready figures

% this part polish some of the figures and make them publication ready
% define all the colors
sFolder = ['..',filesep,'figures'];
figPre = ['placeCell_1D_slice_N200_',date];

nc = 256;   % number of colors
spectralMap = brewermap(nc,'Spectral');
PRGnlMap = brewermap(nc,'PRGn');
% RdBuMap = flip(brewermap(nc,'RdBu'),1);

blues = brewermap(11,'Blues');
reds = brewermap(11,'Reds');
BuGns = brewermap(11,'BuGn');
greens = brewermap(11,'Greens');
greys = brewermap(11,'Greys');
PuRd = brewermap(11,'PuRd');
YlOrRd = brewermap(11,'YlOrRd');
oranges = brewermap(11,'Oranges');
RdBuMap = flip(brewermap(nc,'RdBu'),1);
BlueMap = flip(brewermap(nc,'Blues'),1);
GreyMap = brewermap(nc,'Greys');


% Parameters for plot
figWidth = 3.5;   % in unit of inches
figHeight = 2.8;
plotLineWd = 2;    %line width of plot
axisLineWd = 1;    % axis line width
labelFont = 20;    % font size of labels
axisFont = 16;     % axis font size

% ************************************************************
% Diffusion constant vs the synaptic noise level
% ************************************************************
% plot when we use various noise level for different neurons
if strcmp(noiseVar, 'various')
   
    % first loglog plot
    f_D_sig = figure;
    pos(3)=3.2; pos(4)=2.8;
    set(f_D_sig,'color','w','Units','inches','Position',pos)
    
    loglog(tot_noise_var(eff_acti_inx),Ds(eff_acti_inx),'o','MarkerSize',6,...
        'MarkerFaceColor',greys(1,:),'MarkerEdgeColor',greys(9,:),'LineWidth',1)
  
    
    xlabel('$\sigma_W^2 + \sigma_M^2$','Interpreter','latex','FontSize',16)
    ylabel('$D$','Interpreter','latex','FontSize',16)
    set(gca,'XTick',10.^(-7:-4))

    prefix = [figPre, 'D_sigmas'];
    saveas(f_D_sig,[sFolder,filesep,prefix,'.fig'])
    print('-depsc',[sFolder,filesep,prefix,'.eps'])
  
    % make a linear regression and plot the confidence interval and R^2
    sig_range = 10.^(-7:0.1:-4)';
    D_pred =  predict(mdl,sig_range);
    
    f_D_sig_linear = figure;
    set(f_D_sig_linear,'color','w','Units','inches','Position',pos)
    
    plot(tot_noise_var(eff_acti_inx),Ds(eff_acti_inx),'o','MarkerSize',6,...
       'MarkerEdgeColor',greys(9,:))
    hold on
    plot(sig_range,D_pred,'LineWidth',1.5,'Color',blues(9,:))
    hold off
    xlabel('$\sigma_W^2 + \sigma_M^2$','Interpreter','latex','FontSize',16)
    ylabel('$D$','Interpreter','latex','FontSize',16)
    
    R2 = num2str(round(mdl.Rsquared.Ordinary*100)/100);
    pvalue = num2str(fit_stat.pValue(2));
    annotation('textbox', [.1 .8,.85  .1],...
    'String', ['R^2 =',R2,'; p-value =',pvalue],'BackgroundColor','none','Color','k',...
    'LineStyle','none','fontsize',10,'HorizontalAlignment','Center');

%     prefix = [figPre, 'D_sigmas_linearRegr'];
%     saveas(f_D_sig_linear,[sFolder,filesep,prefix,'.fig'])
%     print('-depsc',[sFolder,filesep,prefix,'.eps'])
    
end

% ************************************************************
% peak amplitude of example neuron
% ************************************************************
inx = 34;  % select one neuron to plot
temp = output.pkAmp(inx,:);
temp(isnan(temp)) = 0;
times = (1:length(temp))*param.step;   % iterations

f_pkAmpTraj = figure;
set(f_pkAmpTraj,'color','w','Units','inches')
pos(3)=3.2;  %17.4; %8.5;%pos(3)*1.5;
pos(4)=2.8;%pos(4)*1.5;
set(f_pkAmpTraj,'Position',pos)

plot(times',temp','LineWidth',2,'Color',blues(9,:))
xlabel('Time','FontSize',16)
ylabel('Peak Amplitude','FontSize',16)
set(gca,'LineWidth',1,'FontSize',16)

% prefix = [figPre, 'pkAmp'];
% saveas(f_pkAmpTraj,[sFolder,filesep,prefix,'.fig'])
% print('-depsc',[sFolder,filesep,prefix,'.eps'])


% ************************************************************
% Fraction of active neurons
% ************************************************************
f_acti = figure;
set(f_acti,'color','w','Units','inches')
pos(3)=3.3;  
pos(4)=2.8;
set(f_acti,'Position',pos)
% actiFrac = mean(pks > 0,1);

plot(times',actiFrac,'LineWidth',1.5)
xlabel('Time','FontSize',16)
ylabel('Active fraction','FontSize',16)
ylim([0,1])
set(gca,'LineWidth',1,'FontSize',16)

% prefix = [figPre, 'fracActive'];
% saveas(f_acti,[sFolder,filesep,prefix,'.fig'])
% print('-depsc',[sFolder,filesep,prefix,'.eps'])


% ************************************************************
% average peak amplitude vs time of active
% ************************************************************

temp = output.pkAmp > ampThd;
tolActiTime = sum(temp,2)/size(output.pkAmp,2);
avePks = nan(size(output.pkAmp));
avePks(temp) = output.pkAmp(temp);
meanPks = nanmean(avePks,2);

f_ampActiTime= figure;
set(f_ampActiTime,'color','w','Units','inches')
pos(3)=3.5;  
pos(4)=2.8;
set(f_ampActiTime,'Position',pos)
plot(meanPks,tolActiTime,'o','MarkerSize',4,...
    'MarkerEdgeColor',greys(9,:),'LineWidth',1.5)
xlabel('Mean peak amplitude','FontSize',labelFont)
% ylabel({'faction of', 'active time'},'FontSize',labelFont)
ylabel(['faction of ', 'active time'],'FontSize',labelFont)
set(gca,'FontSize',axisFont,'LineWidth',axisLineWd)

% prefix = [figPre, 'pkAmp_actiTime'];
% saveas(f_ampActiTime,[sFolder,filesep,prefix,'.fig'])
% print('-depsc',[sFolder,filesep,prefix,'.eps'])


% ***************************************************************
% Change of population vectors, measured by Pearson's correlation
% ***************************************************************
% fmSel = randperm(size(Yt,2),1); % randomly slect three frames to consider
pvCorr = zeros(size(Yt,3),size(Yt,2)); 
% [~,neuroInx] = sort(peakInx(:,inxSel(1)));

for i = 1:size(Yt,3)
    for j = 1:size(Yt,2)
        temp = Yt(:,j,i);
        C = corrcoef(temp,Yt(:,j,1));
        pvCorr(i,j) = C(1,2);
    end
end

f_pvCorr = figure;
set(f_pvCorr,'color','w','Units','inches')
pos(3)=figWidth;  
pos(4)=figHeight;
set(f_pvCorr,'Position',pos)
fh = shadedErrorBar((1:size(pvCorr,1))',pvCorr',{@mean,@std});
box on
set(fh.edge,'Visible','off')
fh.mainLine.LineWidth = 3;
fh.mainLine.Color = blues(10,:);
fh.patch.FaceColor = blues(7,:);
% ylim([0.25,1])
xlim([0,400])
xlabel('Time','FontSize',16)
ylabel('PV correlation','FontSize',16)
set(gca,'FontSize',16,'LineWidth',1)

% prefix = [figPre, 'pvCorrCoef'];
% saveas(f_pvCorr,[sFolder,filesep,prefix,'.fig'])
% print('-depsc',[sFolder,filesep,prefix,'.eps'])

% ************************************************************
% Representational similarity across time
% ************************************************************
% for better viusalization, only use part of the data
% gray color maps
% selInx = 1:10:1024;
selInx = find(~isnan(pks(:,1)));

% Y1= Yt(selInx,:,1);
% Y2 = Yt(selInx,:,1000);
Y1= Yt(:,:,1000);
Y2 = Yt(:,:,1300);

SM1 = Y1'*Y1;
SM2 = Y2'*Y2;

f_sm1 = figure;
set(f_sm1,'color','w','Units','inches')
pos(3)=3;  
pos(4)=3;
set(f_sm1,'Position',pos)

imagesc(SM1,[0,15]);
% colormap(GreyMap)
% cb = colorbar;
% set(cb,'FontSize',12)
set(gca,'XTick','','YTick','','LineWidth',0.5,'Visible','off')

prefix = [figPre, 'sm1'];
saveas(f_sm1,[sFolder,filesep,prefix,'.fig'])
print('-depsc',[sFolder,filesep,prefix,'.eps'])

f_sm2 = figure;
set(f_sm2,'color','w','Units','inches')
pos(3)=3;  
pos(4)=3;
set(f_sm2,'Position',pos)

imagesc(SM2,[0,15]);
set(gca,'XTick','','YTick','','LineWidth',0.5,'Visible','off')

% prefix = [figPre, 'sm2'];
% saveas(f_sm2,[sFolder,filesep,prefix,'.fig'])
% print('-depsc',[sFolder,filesep,prefix,'.eps'])

% ****************************************************************
% heatmap of population vector odered by Y1
% ****************************************************************
[pkSorted,sortedInx] = sort(Y1,2,'descend');
actiTime1Inx = find(pkSorted(:,1)>0.05); % significant nonzeor at time 1
[~,neurOrder] = sort(sortedInx(actiTime1Inx,1));


pv1Fig = figure;
pos(3)=3;  
pos(4)=3;
set(pv1Fig,'color','w','Units','inches','Position',pos)
imagesc(Y1(actiTime1Inx(neurOrder),:),[0,3])
set(gca,'Visible','off')
% colorbar
% prefix = [figPre, 'pvHeatMap1'];
% saveas(pv1Fig,[sFolder,filesep,prefix,'.fig'])
% print('-depsc',[sFolder,filesep,prefix,'.eps'])

pv2Fig = figure;
pos(3)=3;  
pos(4)=3;
set(pv2Fig,'color','w','Units','inches','Position',pos)
imagesc(Y2(actiTime1Inx(neurOrder),:),[0,3])
% imagesc(Y2(neurOrder(250:end),:),[0,2])
set(gca,'Visible','off')

% prefix = [figPre, 'pvHeatMap2_'];
% saveas(pv2Fig,[sFolder,filesep,prefix,'.fig'])
% print('-depsc',[sFolder,filesep,prefix,'.eps'])

%% Statistics of shift of RF, Compare with experiment

% This part is intended to compare with ﻿Gonzalez et al Science 2019 paper
% based on unsorted data

psRange = param.ps;  %total length of the linearized track

% tps = [400,800,1500];
tps = [200,400,800];


quantiles = (1:3)/4*param.ps;
probs = nan(length(tps),length(quantiles));
edges = -200:10:200;
numBins = length(edges)-1; % this is the same as in the experiments

fractionSep = nan(length(tps),numBins);  % store the fraction of different shift
randSiftFrac = nan(length(tps),numBins);  % store randomized shift

shifDist = figure; 
set(shifDist,'color','w','Units','inches')
pos(3)=figWidth;  %17.4; %8.5;%pos(3)*1.5;
pos(4)=figHeight;%pos(4)*1.5;
set(shifDist,'Position',pos)
hold on
for i = 1:length(tps)
    temp = output.pkCenterMass(:,tps(i)+1:end,:) - output.pkCenterMass(:,1:end-tps(i),:);
    
    [f,xi] = ksdensity(temp(~isnan(temp))); 
%     histogram(temp(~isnan(temp)),'Normalization','pdf')
    plot(xi,f,'Color',blues(1+3*i,:),'LineWidth',2)
%     diffLag = min(pi, abs(temp(~isnan(temp))));
    diffLag = abs(temp(~isnan(temp)));
    
   [fractionSep(i,:),~]= histcounts(temp(~isnan(temp)),edges,'Normalization','probability');
    
    for j = 1:length(quantiles)
        probs(i,j) = sum(diffLag(:)>quantiles(j))/length(diffLag(:));
    end
end

% compared with random position
randPks = nan(size(output.pkCenterMass));
for i = 1:size(output.pkCenterMass,2)
    inx = find(~isnan(output.pkCenterMass(:,i)));
    randPks(inx,i) = psRange*rand(length(inx),1);
end
temp = output.pkCenterMass - randPks;
diffLagRnd = abs(temp(~isnan(temp)));
[f,xi] = ksdensity(temp(~isnan(temp))); 


% randomized shift
for i=1:length(tps)
    temp = cell(1,1);
    inx = find(~isnan(output.pkCenterMass(:,tps(i))));
    for j = 1:50
        rps = nan(size(output.pkCenterMass,1),1);
        rps(inx) = psRange*rand(length(inx),1);
        temp{j} = output.pkCenterMass(:,tps(i)) - rps;
    end
    rndSf = cat(1,temp{:});
    [randSiftFrac(i,:),~] = histcounts(rndSf(~isnan(rndSf)),edges,'Normalization','probability');
end

% [randSiftFrac,~] = histcounts(temp(~isnan(temp)),numBins,'Normalization','probability');
plot(xi,f,'Color',reds(8,:),'LineWidth',2)
hold off
box on
% xlim([-6.5,6.5])
lg = legend(['\Delta t=',num2str(tps(1))],['\Delta t=',num2str(tps(2))],...
    ['\Delta t=',num2str(tps(3))],'Random','Location','northeast');
set(lg,'FontSize',14)
xlabel('$\Delta s$','Interpreter','latex','FontSize',24)
ylabel('pdf','FontSize',24)
set(gca,'LineWidth',1,'FontSize',20)

% *****************************************************
% Histount of the probability, compared with experiment
% *****************************************************
histFig = figure;
pos(3)=figWidth; 
pos(4)=figHeight;
set(histFig,'color','w','Units','inches','Position',pos)

hold on
for i = 1:length(tps)
    plot(fractionSep(i,:)','Color',blues(1+3*i,:),'LineWidth',plotLineWd)
    plot(randSiftFrac(i,:)','Color',reds(1+3*i,:),'LineWidth',plotLineWd)
end
box on
lg = legend(['\Delta t=',num2str(tps(1))],['\Delta t=',num2str(tps(2))],...
    ['\Delta t=',num2str(tps(3))],'Random');
set(lg,'FontSize',14)
% xlim([0,50])
xticks([1,(1:4)*numBins/4])
x = -1:0.5:1;
xticklabels({x});
xlabel('Centroid shift','FontSize',labelFont)
ylabel('Fraction','FontSize',labelFont)
set(gca,'FontSize',axisFont)

% save the figure
% prefix = [figPre, 'shiftDistrFrac'];
% saveas(histFig,[sFolder,filesep,prefix,'.fig'])
% print('-depsc',[sFolder,filesep,prefix,'.eps'])
