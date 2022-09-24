% Make figure S2: Ring model with single neuron
% REQUIREMNT:
% run 'ringPlaceModel.m' first to generate the data for panel A & B
% run 'runRingModeDriftComp.m' to generate data for C: data stored in the
% folder "pcRing_centerRW"
% run 'ring_model_three_phases.m' to generate data for D
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
lineWd = 1.5;    %line width of plot
axisLineWd = 1;    % axis line width
labelSize = 20;    % font size of labels
axisFont = 16;     % axis font size
symbSize = 4;

pos = [0 0 3.2 2.8];
%% Fig S2B D vs learning rate

dFolder = '../data_in_paper/';
dFileLR = fullfile(dFolder,'pc1D_ring_learnRate_alp0_Np1_std0_lr0.01_bs1.mat');
load(dFileLR)

figPre = 'pc_ring_N1';    % depending on the noise level, this should be different

% repeats = size(allAmp,2);

aveAmp = mean(meanAmp,2);
stdAmp = std(meanAmp,0,2);

% diffusion constants
aveD = mean(squeeze(allDs/2),1);
stdD = std(squeeze(allDs/2),0,1);  % exact diffusion constant


% **********************************************
% D vs learn rate
% **********************************************
% fit a scaling function
% xs = [ones(length(lrs),1),log10(lrs')];
% bs = xs\log10(aveD');
% y_pred = 10.^(xs*[bs(1);2]);

f_lr_D= figure;
hold on
set(f_lr_D,'color','w','Units','inches','Position',pos)

errorbar(lrs',aveD,stdD,'o','MarkerSize',symbSize,'MarkerFaceColor',...
    greys(9,:),'MarkerEdgeColor',greys(9,:),'Color',greys(9,:),'LineWidth',...
    lineWd,'CapSize',0)
% plot(lrs',y_pred,'Color',PuRd(7,:),'LineWidth',lineWd)
y_pred = lrs.^2/2;
plot(lrs',y_pred','Color',PuRd(7,:),'LineWidth',lineWd)
hold off
box on
lg = legend('simulation','theory','Location','northwest');
% lg = legend('simulation','$D \propto \eta^2$','Location','northwest');
set(lg,'Interpreter','latex','FontSize',12)
xlabel('$\eta$','Interpreter','latex','FontSize',labelSize)
ylabel('$D$','Interpreter','latex','FontSize',labelSize)
set(gca,'FontSize',axisFont,'LineWidth',axisLineWd,'XScale','log','YScale','log',...
    'YTick',10.^(-6:2:-2),'XTick',10.^(-3:1:-1))

%% Fig S2C   AD vs sigma for single neuron in the ring model
dFolder = '../data_in_paper/';
dFileSig = fullfile(dFolder,'pc1D_ring_sigma_alp0_Np1_std0_lr0.01_bs1.mat');  
load(dFileSig)

figPre = 'placeCell_ring_';    % depending on the noise level, this should be different
aveAmp = nanmean(meanAmp,2);
stdAmp = nanstd(meanAmp,0,2);

% diffusion constants
aveD = nanmean(squeeze(allDs/2),1);   % this is the exact value, 8/22/2022
stdD = nanstd(squeeze(allDs/2),0,1);

% theory
lr = 0.01;
Dtheory = lr^2/16 + lr*sigs'.^2./aveAmp.^2;
const = aveD(1)/Dtheory(1);

fitScaling = 0;  % 1 for fit a power law, 0 for our theory
%fit a scaling function
if fitScaling
    xs = [ones(length(sigs),1),log10(sigs')];
    bs = xs\log10(aveD');
    y_pred = 10.^(xs*bs);
else
% fit our theory
    fitFun = @(c) c*log10(Dtheory) - log10(aveD);
    c0 = aveD(1)/Dtheory(1);
    c_best = lsqnonlin(fitFun,c0);
    y_pred = 10.^(c_best*log10(Dtheory));
end

f_sig_D= figure;
hold on
set(f_sig_D,'color','w','Units','inches','Position',pos)

ebh= errorbar(sigs',aveD,stdD,'o','MarkerSize',symbSize,'MarkerFaceColor',...
    greys(9,:),'MarkerEdgeColor',greys(9,:),'Color',greys(9,:),'LineWidth',...
    lineWd,'CapSize',0);

% exact theory, revised on 8/22/2021
x_sig = 10.^(-4:0.05:-1)';
plot(x_sig,(lr^2 + 16*lr*x_sig.^2)/2,'Color',PuRd(7,:),'LineWidth',lineWd)
hold off
box on
lg = legend('simulation','theory','Location','northwest');
set(lg,'Interpreter','latex','FontSize',16)
xlabel('$\sigma$','Interpreter','latex','FontSize',labelSize)
ylabel('$D$','Interpreter','latex','FontSize',labelSize)
set(gca,'FontSize',axisFont,'LineWidth',axisLineWd,'XScale','log','YScale','log',...
    'YTick',10.^(-5:1:-3),'XTick',10.^(-4:1:-1))

%% Fig S2D
dFolder = '../data_in_paper/pcRing_paper_alpha/';
dFileAlp1 = fullfile(dFolder,'pc1D_ring_alpha_alp0_Np1_std0.01_lr0.05_bs1.mat');
datatSet1 = load(dFileAlp1);
dFileAlp2 = fullfile(dFolder,'pc1D_ring_alpha_alp0_Np1_std0.05_lr0.01_bs1.mat');
datatSet2 = load(dFileAlp2);
dFileAlp3 = fullfile(dFolder,'pc1D_ring_alpha_alp0_Np1_std0.01_lr0.01_bs1.mat');
datatSet3 = load(dFileAlp3);

lr1 = 0.05; sig1 = 0.01;
lr2 = 0.01; sig2 = 0.05;
lr3 = 0.01; sig3 = 0.01;


% repeats = size(allAmp,2);
aveAmps = {mean(datatSet1.meanAmp,2);mean(datatSet2.meanAmp,2);...
    mean(datatSet3.meanAmp,2)};
allDs = {squeeze(datatSet1.allDs);squeeze(datatSet2.allDs);squeeze(datatSet3.allDs)};

% **********************************************
% alpha vs amplitude
% **********************************************
blues = brewermap(11,'Blues');
greys = brewermap(11,'Greys');
oranges = brewermap(11,'Oranges');
plotColors = {blues;oranges;greys};

f_D_amp= figure;
set(f_D_amp,'color','w','Units','inches','Position',[0,0,5,2.8])
hold on
for i = 1:3
    fh = shadedErrorBar(aveAmps{i},allDs{i},{@mean,@std});
    set(fh.edge,'Visible','off')
    fh.mainLine.LineWidth = 2;
    fh.mainLine.Color = plotColors{i}(10,:);
    fh.patch.FaceColor = plotColors{i}(7,:);
end
hold off
box on
lg = legend('\eta = 0.05; \sigma = 0.01','\eta = 0.01; \sigma = 0.05',...
    '\eta = 0.01; \sigma = 0.01','Location','eastoutside');
set(lg,'FontSize',12)

set(gca,'FontSize',16,'LineWidth',1,'YScale','log')
xlabel('Amplitude','FontSize',20)
ylabel('$D$','Interpreter','latex','FontSize',20)

% prefix = ['pc_ring_N1_', 'pkAmp_D_lrs_sigs'];
% saveas(gcf,[sFolder,filesep,prefix,'.fig'])
% print('-depsc',[sFolder,filesep,prefix,'.eps'])
