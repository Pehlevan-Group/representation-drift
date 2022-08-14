% Make figure 4: prediction of ring model
% REQUIREMNT:
% run 'ringPlaceModel.m' first to generate the data for panel A & B
% run 'runRingModel.m' first to generate data for panel C
% run  'ringDifferentLearnRate.m' to generat data for panel D
% run 'runRingModeDriftComp.m' to generate data for E: data stored in the
% folder "pcRing_centerRW"
% run 'ring_model_three_phases.m' to generate data for F
%

%% Graphics settings

sFolder = '../figures';
figPre = 'placeRing_Centroid_RW';

nc = 256;   % number of colors
spectralMap = flip(brewermap(nc,'Spectral'));
PRGnlMap = brewermap(nc,'PRGn');
% RdBuMap = flip(brewermap(nc,'RdBu'),1);

blues = brewermap(11,'Blues');
BuGns = brewermap(11,'BuGn');
greens = brewermap(11,'Greens');
greys = brewermap(11,'Greys');
PuRd = brewermap(11,'PuRd');
YlOrRd = brewermap(11,'YlOrRd');
oranges = brewermap(11,'Oranges');
RdBuMap = flip(brewermap(nc,'RdBu'),1);
BlueMap = flip(brewermap(nc,'Blues'),1);
GreyMap = brewermap(nc,'Greys');


figWidth = 3.2;
figHeight = 2.8;
plotLineWd = 2;    %line width of plot
axisLineWd = 1;    % axis line width
labelFont = 20;    % font size of labels
axisFont = 16;     % axis font size

lineWd = 1.5;
symbSize = 4;

% figure size, weight and height
pos(3)=figWidth;  
pos(4)=figHeight;

%% Diffusion constants vs active time, peak amplitude, Fig 4A
% this part should be run either by run 'ringPlaceModel.m' first or by
% loading data generated by 'ringPlaceModel.m'

% load data
dFolder = '../data/batch_0613.mat';
load(dFolder,'tolActiTime','Ds','meanPks')


% ************************************************************
% Diffusion constant and active period
% ************************************************************
timeAmpFig = figure;
pos(3)=3.5;  
pos(4)=2.8;
set(timeAmpFig,'color','w','Units','inches','Position',pos)
plot(meanPks,tolActiTime,'o','MarkerSize',4,...
    'MarkerEdgeColor',greys(9,:),'LineWidth',1.5)
ylabel('Fraction of active time','FontSize',labelFont)
xlabel('Mean peak amplitude','FontSize',labelFont)
set(gca,'FontSize',axisFont,'LineWidth',axisLineWd)
% ylim([0,2.5])

% prefix = [figPre, 'active_diffCont_'];
% saveas(timeAmpFig,[sFolder,filesep,prefix,'.fig'])
% print('-depsc',[sFolder,filesep,prefix,'.eps'])



% ************************************************************
% Diffusion constant vs averaged peak values, Fig 4B
% ************************************************************
DiffuFig = figure;
pos(3)=3.5;  
pos(4)=2.8;
set(DiffuFig,'color','w','Units','inches','Position',pos)
plot(meanPks,Ds(:,1),'o','MarkerSize',4,...
    'MarkerEdgeColor',greys(9,:),'LineWidth',1.5)
xlabel('Mean peak amplitude','FontSize',labelFont)
ylabel('$D$','Interpreter','latex','FontSize',labelFont)
set(gca,'YTick',10.^(-4:-1),'FontSize',axisFont,'LineWidth',axisLineWd,'YScale','log')
% ylim([0,2.5])

% prefix = [figPre, 'ampl_diffCont_'];
% saveas(DiffuFig,[sFolder,filesep,prefix,'.fig'])
% print('-depsc',[sFolder,filesep,prefix,'.eps'])

%% Diffusion contant vs learning rate


% ************************************************************
% Diffusion constant vs homogeneous learning rate, Fig 4C
% ************************************************************
dFile = '../data/pc1D_ring_learnRate_alp0_Np100_std0.01_lr0.05_bs1-09-Jun-2022.mat';
load(dFile,'allDs','Np','lrs','repeats')

transformedDs = reshape(allDs,[Np*repeats,length(lrs)]);

% aveAmp = mean(meanAmp,2);
% stdAmp = std(meanAmp,0,2);

% diffusion constants
aveD = nanmean(transformedDs,1);
stdD = nanstd(transformedDs,0,1);

f_lr_D= figure;
hold on
set(f_lr_D,'color','w','Units','inches','Position',pos)

ebh = errorbar(lrs',aveD,stdD,'o','MarkerSize',5,'MarkerFaceColor',...
    greys(9,:),'MarkerEdgeColor',greys(9,:),'Color',greys(9,:),'LineWidth',...
    lineWd,'CapSize',0);
ebh.YNegativeDelta = [];
% plot(lrs',y_pred,'Color',PuRd(7,:),'LineWidth',lineWd)
hold off
box on

ylim([2e-4,0.1])
xlim([5e-4,0.2])
xlabel('$\eta$','Interpreter','latex','FontSize',20)
ylabel('$\langle D_i \rangle$','Interpreter','latex','FontSize',20)
set(gca,'FontSize',16,'LineWidth',1,'XScale','log','YScale','log',...
    'YTick',10.^(-4:1:-1),'XTick',10.^(-3:1:-1))

% prefix = 'pcRing_D_lr_noise_001_';
% saveas(f_lr_D,[sFolder,filesep,prefix,'.fig'])
% print('-depsc',[sFolder,filesep,prefix,'.eps'])


% ************************************************************
% individual diffusion constant vs learning rate, Fig. 4D
% ************************************************************
dFile = '../data/ring_diff_learnRate_0608.mat';
load(dFile,'lrs','Ds')

% Ds vs learning rate
f_D_lr= figure;
set(f_D_lr,'color','w','Units','inches','Position',pos)

plot(lrs,Ds,'o','MarkerSize',5,'MarkerEdgeColor',greys(9,:),'Color',greys(9,:),...
    'LineWidth',2)
xlabel('$\eta_i$','Interpreter','latex','FontSize',20)
ylabel('$D_i$','Interpreter','latex','FontSize',20)
xlim([5e-4,1e-1])
ylim([2e-4,1e-1])
set(gca,'FontSize',16,'LineWidth',1,'XScale','log','YScale','log','XTick',10.^(-3:-1),...
    'YTick',10.^(-3:-1))

% prefix = ['ring_different_learnRate_sig',num2str(params.noise),'_3'];
% saveas(f_D_lr,[sFolder,filesep,prefix,'.fig'])
% print('-depsc',[sFolder,filesep,prefix,'.eps'])

%% Compare with random walkers, Fig 4E

% parepare the data
dFolder = '../data/pcRing_centerRW/';
allFile = dir([dFolder,filesep,'*.mat']);
files = {allFile.name}';

s1 = '(?<= *_Np)[\d]+(?=_)';

numNs = 10;    % could be 9 or 10, depending on the dataset
actiNs = nan(numNs,1);   % store the effective Ns in the interacting model
Ns = nan(length(files),1);  % the Np in simulation
varDs = nan(length(files),2);   % store all the mean variance and std of variance

% first, concantenate N > 
for i0 = 1:length(files)
    Ns(i0) = str2num(char(regexp(files{i0},s1,'match')));
    raw = load(char(fullfile(dFolder,filesep,files{i0})));
    
    if raw.Np > 1
        meanVarDs = nan(raw.repeats,1);
        for k = 1:raw.repeats
            meanVarDs(k) = mean(raw.allCentroidDist{k}(:,2));  
        end
        varDs(i0,:) = [mean(meanVarDs),std(meanVarDs)];
        actiNs(i0) = mean(raw.allAveActi);
    end
end

% sort Ns
[~,ix] = sort(Ns);

% plot(varDs(ix))

% sorted variance of centroids
sortedVarDsInteract = varDs(ix(2:end),:);
effNs_sorted = round(actiNs(ix(2:end)));

% mimic random walk on the ring
raw = load(char(fullfile(dFolder,filesep,files{ix(1)})));
allIndependentDs = nan(numNs-1,2);
for i0 = 1:numNs-1
    % randomly select df
    varDs_eachN = nan(40,1);
    for j = 1:40
        trajSel = randperm(100,effNs_sorted(i0));
        mergeTraj = cat(1,raw.allPks{trajSel});
        
        % estimate the centroid distance
        nearestDist = nan(size(mergeTraj,2),2);  % store the mean and std
        for k = 1:size(mergeTraj,2)
            temp = sort(mergeTraj(~isnan(mergeTraj(:,k)),k));
            ds = [diff(temp);(temp(1)-temp(end)) + 2*pi];
            nearestDist(k,:) = [mean(ds),var(ds)];
        end
        varDs_eachN(j) = mean(nearestDist(:,2));
    end
    
    allIndependentDs(i0,:) = [mean(varDs_eachN),std(varDs_eachN)];
end

% PLOT
f_centerComp = figure;
set(f_centerComp,'color','w','Units','inches','Position',pos)
hold on
errorbar(effNs_sorted,allIndependentDs(:,1),allIndependentDs(:,2),'o-','MarkerSize',...
    symbSize,'MarkerFaceColor',greys(9,:),'MarkerEdgeColor',greys(9,:),...
    'Color',greys(9,:),'LineWidth',lineWd,'CapSize',0)
errorbar(effNs_sorted,sortedVarDsInteract(:,1),sortedVarDsInteract(:,2),'o-','MarkerSize',...
    symbSize,'MarkerFaceColor',blues(9,:),'MarkerEdgeColor',blues(9,:),...
    'Color',blues(9,:),'LineWidth',lineWd,'CapSize',0)
box on
legend('independent','model')
xlabel('$N_{\rm{active}}$','Interpreter','latex','FontSize',20)
ylabel('$\langle(\Delta s - \bar{\Delta s})^2 \rangle (\rm{rad}^2)$','Interpreter','latex','FontSize',20)
% ylabel('$\langle(\Delta s - \langle\Delta s\rangle)^2 \rangle (\rm{rad}^2)$','Interpreter','latex','FontSize',20)
% set(gca,'YScale','linear','XScale','linear','FontSize',16)
set(gca,'YScale','log','XScale','linear','YTick',10.^(-2:1:1),'FontSize',16)
% set(gca,'YScale','log','XScale','linear','FontSize',16)
xlim([0,30])

%% Compare different noise sources  Fig. 4F
% Nois is introduced to either feedforward, recurrent or all matrices

% The figure can be generated directly by running
% 'ring_model_three_phases.m' or by loading the data generated
dFile = '../data/ring_model_different_noise.mat';
load(dFile,'total_iter','params','all_Yts')

blues = brewermap(11,'Blues');
reds = brewermap(11,'Reds');
greys = brewermap(11,'Greys');
fig_colors = [blues([7,11],:);reds([7,11],:);greys([7,11],:)];

time_points = round(total_iter/params.record_step);
   
f_pvCorr = figure;
set(f_pvCorr,'color','w','Units','inches')
pos(3)=3.5;  
pos(4)=2.8;
set(f_pvCorr,'Position',pos)

for phase = 1:3
    pvCorr = zeros(size(all_Yts{phase},3),size(all_Yts{phase},2)); 
    for i = 1:size(all_Yts{phase},3)
        for j = 1:size(all_Yts{phase},2)
            temp = all_Yts{phase}(:,j,i);
            C = corrcoef(temp,all_Yts{phase}(:,j,1));
            pvCorr(i,j) = C(1,2);
        end
    end
    PV_corr_coefs{phase} = pvCorr;
    % plot
    fh = shadedErrorBar((1:size(pvCorr,1))'*params.record_step,pvCorr',{@mean,@std});
    box on
    set(fh.edge,'Visible','off')
    fh.mainLine.LineWidth = 3;
    fh.mainLine.Color = fig_colors(2*phase-1,:);
    fh.patch.FaceColor = fig_colors(2*phase,:);
end
lg = legend('Full model','Forward noise', 'Recurrent noise');
set(lg,'FontSize',10)

% ylim([0.25,1])
xlim([0,100]*params.record_step)
xlabel('Time','FontSize',16)
ylabel('PV correlation','FontSize',16)
set(gca,'FontSize',16,'LineWidth',1)