% Figure 3
% The data are from several different simulations


%% graphics settingss

nc = 256;   % number of colors
spectralMap = flip(brewermap(nc,'Spectral'));
PRGnlMap = brewermap(nc,'PRGn');

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
lineWd = 1.5;
symbSize = 4;
labelSize = 20;
axisSize = 16;
axisWd = 1;

% figure size, weight and height
pos(3)=figWidth;  
pos(4)=figHeight;

saveFolder = '../figures';

%% Fig 3A
% dFile = '../data/exampleRF.mat';
dFile = '../data_in_paper/exampleRF.mat';
load(dFile,'Ys','X')

% illustration of ring data manifold
visual_sep = 5;   % only plot very 5 data point
num_angle = size(X,2);
colors = flip(brewermap(round(num_angle/visual_sep),'Spectral'));

ringfig = figure;
set(ringfig,'color','w','Units','inches','Position',[0,0,3.2,2.8])
hold on
for i = 1:floor(num_angle/visual_sep)
    plot(X(1,i*visual_sep),X(2,i*visual_sep),'.','Color',colors(i,:),'MarkerSize',10)
end
hold off
colormap(spectralMap)
cbh = colorbar;
cbh.Ticks = linspace(0, 1, 3) ; %Create 8 ticks from zero to 1
cbh.TickLabels = {'0','\pi','2\pi'} ;
pbaspect([1 1 1])
set(gca,'XColor','none','YColor','none')

% prefix ='ring_input_manifold_06162022';
% saveas(ringfig,[saveFolder,filesep,prefix,'.fig'])
% print('-depsc',[saveFolder,filesep,prefix,'.eps'])

%% Fig 3B
% this is a seperate data set when ran with N = 10 and without noise
sep = 2*pi/size(X,2);
thetas = 0:sep:2*pi-sep;

rfExp = figure;
set(rfExp,'color','w','Units','inches','Position',[0,0,3.2,2.8])
plot(thetas,Ys','LineWidth',2)
xlabel('Position','FontSize',20)
ylabel('Response','FontSize',20)
set(gca,'FontSize',16,'XTick',[0,pi,2*pi-sep],'XTickLabel',{'0','\pi','2\pi'})

% prefix ='ring_RF_example_06162022';
% saveas(rfExp,[saveFolder,filesep,prefix,'.fig'])
% print('-depsc',[saveFolder,filesep,prefix,'.eps'])


%% Example drift RF, Fig 3C
% this part requies new online simulation to make it consistant with 
% dFile = '../data/pcRing_paper/onlineN200_0613.mat';
dFile = '../data_in_paper/ring_N5_noise_06232022.mat';

load(dFile, 'newPeaks','params')
saveFolder = '../figures';

sel = [3,4];
% sel = [52];
centroidSel = newPeaks(sel,:)';
centroidFig = figure;
set(centroidFig,'color','w','Units','inches','Position',[0,0,3.2,2.8])
plot((1:size(centroidSel,1))'*params.record_step,centroidSel,'LineWidth',1.5)
% xlim([0,1e4]);
ylim([0,2*pi])
xlabel('Time','FontSize',20)
ylabel('Centroid of RF','FontSize',20)
set(gca,'FontSize',16,'YTick',[0,pi,2*pi],'YTickLabel',{'0','\pi','2\pi'})

% save the figure
% prefix ='ringPlace_centroid_example_06232022';
% saveas(centroidFig,[saveFolder,filesep,prefix,'.fig'])
% print('-depsc',[saveFolder,filesep,prefix,'.eps'])
% 



%% Fig 3D
% ********************************************
%  Similarity matrix
% ********************************************
dFile = '../data_in_paper/ringPlace_N200_online_06202022.mat';
load(dFile)


% sel = randperm(k,2);
Y1 = Yt(:,:,1);
Y2 = Yt(:,:,1e3);

smFig1 = figure;
set(smFig1,'color','w','Units','inches','Position',[0,0,3.5,2.9])
imagesc(Y1'*Y1,[0,1.2])
% colorbar
title('$t=1$','Interpreter','latex','FontSize',20)
xlabel('Position','FontSize',20)
ylabel('Position','FontSize',20)
set(gca,'FontSize',16,'YTick',[1,500,1000],'YTickLabel',{'0','\pi','2\pi'},...
    'XTick',[1,500,1000],'XTickLabel',{'0','\pi','2\pi'})

% prefix ='ring_sm_example_t1_06192022';
% saveas(smFig1,[saveFolder,filesep,prefix,'.fig'])
% print('-depsc',[saveFolder,filesep,prefix,'.eps'])


smFig2 = figure;
set(smFig2,'color','w','Units','inches','Position',[0,0,3.5,2.9])
imagesc(Y2'*Y2,[0,1.2])
colorbar
title('$t=10^4$','Interpreter','latex','FontSize',20)
xlabel('Position','FontSize',20)
xlabel('Position','FontSize',20)
set(gca,'FontSize',16,'YTick',[1,500,1000],'YTickLabel',{'0','\pi','2\pi'},...
    'XTick',[1,500,1000],'XTickLabel',{'0','\pi','2\pi'})

% prefix ='ring_sm_example_t2';
% saveas(smFig2,[saveFolder,filesep,prefix,'.fig'])
% print('-depsc',[saveFolder,filesep,prefix,'.eps'])\

%% Fig 3E
% ************************************************************
% peak amplitude of example neuron
% ************************************************************
inx = 17;  % select one neuron to plot
temp = newPeakVals(inx,:);
temp(isnan(temp)) = 0;
times = (1:length(temp))*params.record_step;   % iterations

f_pkAmpTraj = figure;
set(f_pkAmpTraj,'color','w','Units','inches')
pos(3)=3.2;  %17.4; %8.5;%pos(3)*1.5;
pos(4)=2.8;%pos(4)*1.5;
set(f_pkAmpTraj,'Position',pos)
xlim([0,1e4])

plot(times',temp','LineWidth',2,'Color',blues(9,:))
xlabel('Time','FontSize',16)
ylabel('Peak Amplitude','FontSize',16)
set(gca,'LineWidth',1,'FontSize',16)

% prefix = ['ringPlace_', 'pkAmp_example_06202022'];
% saveas(f_pkAmpTraj,[saveFolder,filesep,prefix,'.fig'])
% print('-depsc',[saveFolder,filesep,prefix,'.eps'])


%% Fig 3F
% ************************************************************
% Fraction of active neurons
% ************************************************************
f_acti = figure;
set(f_acti,'color','w','Units','inches')
pos(3)=3.3;  
pos(4)=2.8;
set(f_acti,'Position',pos)
actiFrac = mean(newPeaks > 0,1); 

subSample = 5:5:size(newPeaks,2);

plot(subSample'*params.record_step,actiFrac(subSample),'LineWidth',2)
xlabel('Time','FontSize',16)
ylabel('Active fraction','FontSize',16)
ylim([0,1])
set(gca,'LineWidth',1,'FontSize',16)

% prefix = ['ringPlace_', 'fracActive_06202022'];
% saveas(f_acti,[saveFolder,filesep,prefix,'.fig'])
% print('-depsc',[saveFolder,filesep,prefix,'.eps'])

%% Fig 3G
% load the data for no noise scenario but different N
%
dFileN= fullfile('../data','pc1D_ring_Np_alp0_Np1_std0_lr0.05_bs1_0217.mat');  % new simulation

load(dFileN,'actiFrac')

dFolder = '../data/pcRing_Ndp_0224';
figPre = 'pc';
allFile = dir([dFolder,filesep,'*.mat']);
files = {allFile.name}';

s1 = '(?<= *std)[\d\.]+(?=_)';

numNs = 10;    % could be 9 or 10, depending on the dataset
aveDs = nan(length(files),numNs);   % store all the mean Ds
stdDs = nan(length(files),numNs);   % store all the std of Ds
allMaxMSD = nan(length(files),numNs, 2);   % mean and standard
allWidthRF = nan(length(files),numNs, 2);   % mean and standard
allFracActi = nan(length(files),numNs, 2);   % mean and standard
sigs = nan(length(files),1);    % store all the standard deviation
for i0 = 1:length(files)
    sigs(i0) = str2num(char(regexp(files{i0},s1,'match')));
    raw = load(char(fullfile(dFolder,filesep,files{i0})));
    for j = 1:length(raw.Nps)
        if raw.Nps(j) < 50
            temp = cat(2,raw.allDs{j,:});
            aveDs_noise(j,:) = [nanmean(temp(:)),nanstd(temp(:))];
            
            temp = cat(2,raw.allMaxMSD{j,:});
            allMaxMSD(i0,j,1) = nanmean(temp(:));
            allMaxMSD(i0,j,2) = nanstd(temp(:));
            
            temp = cat(2,raw.allRFwidth{j,:});
            allWidthRF(i0,j,1) = nanmean(temp(temp>0));
            allWidthRF(i0,j,2) = nanstd(temp(temp>0));
        else
            aveDs_noise(j,:) = [nanmean(raw.allDs{j,1}),nanstd(raw.allDs{j,1})];
            allMaxMSD(i0,j,1) = nanmean(raw.allMaxMSD{j,1});
            allMaxMSD(i0,j,2) = nanstd(raw.allMaxMSD{j,1});
            
            allWidthRF(i0,j,1) = nanmean(raw.allRFwidth{j,1});
            allWidthRF(i0,j,2) = nanstd(raw.allRFwidth{j,1});
        end
    end
    
    aveDs(i0,:) = aveDs_noise(:,1)';
    stdDs(i0,:) = aveDs_noise(:,2)';
    
    
    allFracActi(i0,:,1) = nanmean(raw.actiFrac,2)';
    allFracActi(i0,:,2) = nanstd(raw.actiFrac,[],2)';
end

[~,Ix] = sort(sigs,'ascend');


actiFrac_fig= figure;
hold on
set(actiFrac_fig,'color','w','Units','inches','Position',pos)
hold on
plot(2.^(0:1:9)',nanmean(actiFrac,2),'LineWidth',2,'Color',greys(4,:))
plot(2.^(0:1:9)',allFracActi(Ix(1),:,1),'LineWidth',2,'Color',greys(7,:))
plot(2.^(0:1:9)',allFracActi(Ix(end),:,1),'LineWidth',2,'Color',greys(10,:))
hold off
lg = legend('\sigma = 0','\sigma = 10^{-4}','\sigma = 10^{-1}' );
set(lg,'FontSize',14)
box on
xlim([1,520])
% xlabel('$\log_2(N)$','Interpreter','latex')
xlabel('$N$','Interpreter','latex','FontSize',20)
ylabel('Fraction of active neurons','FontSize',20)
set(gca,'LineWidth',1,'FontSize',16,'XTick',[100,300,500])



%% This is part of figure 4 A,B

aveAmp = nanmean(newPeakVals,2);
tolActiTime = nanmean(newPeakVals>0,2);

% ************************************************************
% Diffusion constant and active period
% ************************************************************
timeAmpFig = figure;
pos(3)=3.5;  
pos(4)=2.8;
set(timeAmpFig,'color','w','Units','inches','Position',pos)
plot(aveAmp,tolActiTime,'o','MarkerSize',6,...
    'MarkerEdgeColor',greys(9,:),'LineWidth',1.5)
xlim([0.1,0.45]);ylim([-0.02,1.03])
ylabel('Fraction of active time','FontSize',20)
xlabel('Mean peak amplitude','FontSize',20)
set(gca,'FontSize',16,'LineWidth',1.5)

% prefix = ['fingPlace_', 'active_diffCont_06202022'];
% saveas(timeAmpFig,[saveFolder,filesep,prefix,'.fig'])
% print('-depsc',[saveFolder,filesep,prefix,'.eps'])

% ************************************************************************
%  Diffusion constant vs average amplitude, Figure 4B
% ************************************************************************
fAmpD = figure;
set(fAmpD,'color','w','Units','inches','Position',[0,0,3.5,2.8])
plot(nanmean(newPeakVals,2),Ds_log,'o','MarkerSize',6,'MarkerEdgeColor',greys(9,:),...
    'LineWidth',1.5)
xlabel('Mean peak amplitude','FontSize',20)
ylabel('$D$','Interpreter','latex','FontSize',20)
xlim([0.1,0.45])
ylim([2e-3,2e-1])
set(gca,'YScale','log','YTick',10.^(-3:-1),'FontSize',16,'LineWidth',1.5,'YScale','log')



% linear regression
% mdl = fitlm(aveAmp,log10(Ds_log));
% fit_stat = anova(mdl,'summary');

% prefix = ['ringPlace_', 'ampl_diffCont_06202022'];
% saveas(fAmpD,[saveFolder,filesep,prefix,'.fig'])
% print('-depsc',[saveFolder,filesep,prefix,'.eps'])