% plot the figure in the 1D ring place cell model
% 

close all
clear

%% Graphics settings
sFolder = '../figures';
figPre = 'placeCell_ring_0224';

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
lineWd = 1.5;
symbSize = 4;
labelSize = 20;
axisSize = 16;
axisWd = 1;

% figure size, weight and height
pos(3)=figWidth;  
pos(4)=figHeight;

dFolder = '../data_in_paper/pcRing_paper_alpha/';


%% ampltidue and alpha, diffusion constants, 1 place cell

dFileAlp = fullfile(dFolder,'pc1D_ring_alpha_alp0_Np1_std0.01_lr0.05_bs1.mat');
load(dFileAlp)

lr = 0.05;
sig = 0.01;

% repeats = size(allAmp,2);

aveAmp = mean(meanAmp,2);
stdAmp = std(meanAmp,0,2);

% diffusion constants
aveD = mean(squeeze(allDs),1);
stdD = std(squeeze(allDs),0,1);

% **********************************************
% alpha vs amplitude
% **********************************************
f_alp_amp= figure;
set(f_alp_amp,'color','w','Units','inches','Position',pos)

errorbar(alps',aveAmp,stdAmp,'-o','MarkerSize',symbSize,'MarkerFaceColor',...
    greys(9,:),'MarkerEdgeColor',greys(9,:),'Color',greys(9,:),'LineWidth',...
    lineWd,'CapSize',0)
xlabel('$\alpha^2$','Interpreter','latex','FontSize',labelSize)
ylabel('Amplitude','FontSize',labelSize)
set(gca,'FontSize',axisSize,'LineWidth',axisWd)

% prefix = [figPre, 'pkAmp_alpha_lr',num2str(lr),'_sig',num2str(sig)];
% saveas(f_alp_amp,[sFolder,filesep,prefix,'.fig'])
% print('-depsc',[sFolder,filesep,prefix,'.eps'])


% **********************************************
% D vs alpha
% **********************************************
f_alp_D= figure;
set(f_alp_D,'color','w','Units','inches','Position',pos)

errorbar(alps',aveD,stdD,'-o','MarkerSize',symbSize,'MarkerFaceColor',...
    greys(9,:),'MarkerEdgeColor',greys(9,:),'Color',greys(9,:),'LineWidth',...
    lineWd,'CapSize',0)
xlabel('$\alpha^2$','Interpreter','latex','FontSize',labelSize)
ylabel('$D$','Interpreter','latex','FontSize',labelSize)
set(gca,'FontSize',axisSize,'LineWidth',axisWd,'YScale','log')

% prefix = [figPre, 'pk_alphas_D_lr',num2str(lr),'_sig',num2str(sig)];
% saveas(f_alp_D,[sFolder,filesep,prefix,'.fig'])
% print('-depsc',[sFolder,filesep,prefix,'.eps'])


% **********************************************
% D vs ampltitude
% **********************************************
f_amp_D= figure;
set(f_amp_D,'color','w','Units','inches','Position',pos)

errorbar(aveAmp,aveD,stdD,'-o','MarkerSize',symbSize,'MarkerFaceColor',...
    greys(9,:),'MarkerEdgeColor',greys(9,:),'Color',greys(9,:),'LineWidth',...
    lineWd,'CapSize',0)
xlabel('Amplitude','FontSize',labelSize)
ylabel('$D$','Interpreter','latex','FontSize',labelSize)
set(gca,'FontSize',axisSize,'LineWidth',axisWd,'YScale','log')

% prefix = [figPre, 'pkAmp_D_lr',num2str(lr),'_sig',num2str(sig)];
% saveas(f_amp_D,[sFolder,filesep,prefix,'.fig'])
% print('-depsc',[sFolder,filesep,prefix,'.eps'])



% fit with theory
% aveWidth = nan(size(allRFwidth,1));
lbd = 0.05;
temp = cell2mat(allRFwidth);
temp(temp==0) = nan;
psi = nanmean(temp,2)*pi;   % psi
mu = aveAmp./(1-cos(psi));
% mu = sqrt((2*psi - sin(2*psi)-4*lbd*pi)./(4*psi + 2*psi.*cos(2*psi) - 3*sin(2*psi)));
mu_hat = mu.^(2*psi - sin(psi))/4/pi;
% amp = mu.*(1-cos(psi));
% alp2 = cos(psi).*(2*psi - sin(2*psi))./4./(sin(psi) - psi.*cos(psi));

gammas = pi/6*(36*psi + 24*psi.*cos(2*psi) - 28*sin(2*psi) - sin(4*psi))./(2*psi-sin(2*psi)).^2;
Ds_theory = gammas.*lr.^2 + lr.*sig^2./mu_hat.^2;


figure
plot(aveAmp,Ds_theory)
set(gca,'YScale','linear')

%% Merge the data from two parameter sets into one
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
set(f_D_amp,'color','w','Units','inches','Position',[0,0,4,3])
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

%% Learning rate dependence, no noise
% dFileLR = fullfile(dFolder,'pc1D_ring_learnRate_alp0_Np1_std0_lr0.01.mat');
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
% amplitude vs learn rate
% **********************************************
f_lr_amp= figure;
set(f_lr_amp,'color','w','Units','inches','Position',pos)

errorbar(lrs',aveAmp,stdAmp,'-o','MarkerSize',symbSize,'MarkerFaceColor',...
    greys(9,:),'MarkerEdgeColor',greys(9,:),'Color',greys(9,:),'LineWidth',...
    lineWd,'CapSize',0)
xlabel('$\eta$','Interpreter','latex','FontSize',labelSize)
ylabel('Amplitude','FontSize',labelSize)
ylim([0.5,0.6])
set(gca,'FontSize',axisSize,'LineWidth',axisWd,'XScale','log','XTick',10.^(-3:1:-1))

% prefix = [figPre, 'pkAmp_lr_0217'];
% saveas(f_lr_amp,[sFolder,filesep,prefix,'.fig'])
% print('-depsc',[sFolder,filesep,prefix,'.eps'])


% **********************************************
% D vs learn rate
% **********************************************
% fit a scaling function
xs = [ones(length(lrs),1),log10(lrs')];
bs = xs\log10(aveD');
y_pred = 10.^(xs*[bs(1);2]);

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
set(gca,'FontSize',axisSize,'LineWidth',axisWd,'XScale','log','YScale','log',...
    'YTick',10.^(-6:2:-2),'XTick',10.^(-3:1:-1))
% 
% prefix = [figPre, 'D_lr_0217'];
% saveas(f_lr_D,[sFolder,filesep,prefix,'.fig'])
% print('-depsc',[sFolder,filesep,prefix,'.eps'])


%% Learning rate dependence, with noise

dFileLR = fullfile(dFolder,'pc1D_ring_learnRate_alp0_Np1_std0_lr0.01_bs1.mat');
load(dFileLR)

figPre = 'placeCell_ring_sig0.01';    % depending on the noise level, this should be different


aveAmp = mean(meanAmp,2);
stdAmp = std(meanAmp,0,2);

% diffusion constants
aveD = nanmean(squeeze(allDs),1);
stdD = nanstd(squeeze(allDs),0,1);

% theory
sig = 0.01;
Dtheory = lrs'.^2/16 + lrs'.*sig^2./aveAmp.^2;

const = aveD(1)/Dtheory(1);
% **********************************************
% amplitude vs learn rate
% **********************************************
f_lr_amp= figure;
set(f_lr_amp,'color','w','Units','inches','Position',pos)

errorbar(lrs',aveAmp,stdAmp,'-o','MarkerSize',symbSize,'MarkerFaceColor',...
    greys(9,:),'MarkerEdgeColor',greys(9,:),'Color',greys(9,:),'LineWidth',...
    lineWd,'CapSize',0)
xlabel('$\alpha^2$','Interpreter','latex','FontSize',labelSize)
ylabel('Amplitude','FontSize',labelSize)
ylim([0.5,0.6])
set(gca,'FontSize',axisSize,'LineWidth',axisWd)

% prefix = [figPre, 'pkAmp_lr_noise'];
% saveas(f_lr_amp,[sFolder,filesep,prefix,'.fig'])
% print('-depsc',[sFolder,filesep,prefix,'.eps'])


% **********************************************
% D vs learn rate
% **********************************************
fitScaling = 1;  % 1 for fit a power law, 0 for our theory
%fit a scaling function
if fitScaling
    xs = [ones(length(lrs),1),log10(lrs')];
    bs = xs\log10(aveD');
    y_pred = 10.^(xs*bs);
else
% fit our theory
    fitFun = @(c) c*Dtheory - aveD;
    c0 = aveD(1)/Dtheory(1);
    c_best = lsqnonlin(fitFun,c0);
    y_pred = c_best*Dtheory;
end

f_lr_D= figure;
hold on
set(f_lr_D,'color','w','Units','inches','Position',pos)

errorbar(lrs',aveD,stdD,'o','MarkerSize',symbSize,'MarkerFaceColor',...
    greys(9,:),'MarkerEdgeColor',greys(9,:),'Color',greys(9,:),'LineWidth',...
    lineWd,'CapSize',0)
plot(lrs',y_pred,'Color',PuRd(7,:),'LineWidth',lineWd)
hold off
box on
% lg = legend('simulation','$D \propto \eta^2/16 + \eta\sigma^2/\mu^2$','Location','northwest');
lg = legend('simulation',['$D \propto \eta^{',num2str(round(bs(2)*100)/100),'}$'],'Location','northwest');
set(lg,'Interpreter','latex','FontSize',12)
xlabel('$\eta$','Interpreter','latex','FontSize',labelSize)
ylabel('$D$','Interpreter','latex','FontSize',labelSize)
set(gca,'FontSize',axisSize,'LineWidth',axisWd,'XScale','log','YScale','log',...
    'YTick',10.^(-6:2:-2),'XTick',10.^(-3:1:-1))

% prefix = [figPre, 'D_lr_noise'];
% saveas(f_lr_D,[sFolder,filesep,prefix,'.fig'])
% print('-depsc',[sFolder,filesep,prefix,'.eps'])


%% Learning rate dependent diffusion constants, 100 neurons

dFileLR = fullfile('../data_in_paper', 'pc1D_ring_learnRate_alp0_Np100_std0.01_lr0.05_bs1-09-Jun-2022.mat');

load(dFileLR)

figPre = 'placeCell_ring_N100';    % depending on the noise level, this should be different

transformedDs = reshape(allDs,[Np*repeats,length(lrs)]);

aveAmp = mean(meanAmp,2);
stdAmp = std(meanAmp,0,2);

% diffusion constants
aveD = nanmean(transformedDs,1);
stdD = nanstd(transformedDs,0,1);

% theory
sig = 0.01;
Dtheory = lrs'.^2/16 + lrs'.*sig^2./aveAmp.^2;

const = aveD(1)/Dtheory(1);
% **********************************************
% amplitude vs learn rate
% **********************************************
f_lr_amp= figure;
set(f_lr_amp,'color','w','Units','inches','Position',pos)

errorbar(lrs',aveAmp,stdAmp,'-o','MarkerSize',symbSize,'MarkerFaceColor',...
    greys(9,:),'MarkerEdgeColor',greys(9,:),'Color',greys(9,:),'LineWidth',...
    lineWd,'CapSize',0)
xlabel('$\alpha^2$','Interpreter','latex','FontSize',labelSize)
ylabel('Amplitude','FontSize',labelSize)
ylim([0.5,0.6])
set(gca,'FontSize',axisSize,'LineWidth',axisWd)

prefix = [figPre, 'pkAmp_lr_noise'];
saveas(f_lr_amp,[sFolder,filesep,prefix,'.fig'])
print('-depsc',[sFolder,filesep,prefix,'.eps'])


% **********************************************
% D vs learn rate
% **********************************************
fitScaling = 1;  % 1 for fit a power law, 0 for our theory
%fit a scaling function
if fitScaling
    xs = [ones(length(lrs),1),log10(lrs')];
    bs = xs\log10(aveD');
    y_pred = 10.^(xs*bs);
else
% fit our theory
    fitFun = @(c) c*Dtheory - aveD;
    c0 = aveD(1)/Dtheory(1);
    c_best = lsqnonlin(fitFun,c0);
    y_pred = c_best*Dtheory;
end

f_lr_D= figure;
hold on
set(f_lr_D,'color','w','Units','inches','Position',pos)

ebh = errorbar(lrs',aveD,stdD,'o','MarkerSize',symbSize,'MarkerFaceColor',...
    greys(9,:),'MarkerEdgeColor',greys(9,:),'Color',greys(9,:),'LineWidth',...
    lineWd,'CapSize',0);
ebh.YNegativeDelta = [];
% plot(lrs',y_pred,'Color',PuRd(7,:),'LineWidth',lineWd)
hold off
box on
% lg = legend('simulation','$D \propto \eta^2/16 + \eta\sigma^2/\mu^2$','Location','northwest');
% lg = legend('simulation',['$D \propto \eta^{',num2str(round(bs(2)*100)/100),'}$'],'Location','northwest');
% set(lg,'Interpreter','latex','FontSize',12)
ylim([2e-4,0.1])
xlim([5e-4,0.2])
xlabel('$\eta$','Interpreter','latex','FontSize',labelSize)
ylabel('$\langle D_i \rangle$','Interpreter','latex','FontSize',labelSize)
set(gca,'FontSize',axisSize,'LineWidth',axisWd,'XScale','log','YScale','log',...
    'YTick',10.^(-4:1:-1),'XTick',10.^(-3:1:-1))

% prefix = 'pcRing_D_lr_noise_001_';
% saveas(f_lr_D,[sFolder,filesep,prefix,'.fig'])
% print('-depsc',[sFolder,filesep,prefix,'.eps'])

%}
%% Diffusion constants vs noise amplitude, single cell
% dFileSig = fullfile(dFolder,'pc1D_ring_sigma_alp0_Np1_std0_lr0.01.mat');  % previous simulation
dFolder = '../data_in_paper/';
dFileSig = fullfile(dFolder,'pc1D_ring_sigma_alp0_Np1_std0_lr0.01_bs1.mat');  % previous simulation

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
% **********************************************
% amplitude vs learn rate
% **********************************************
f_sig_amp= figure;
set(f_sig_amp,'color','w','Units','inches','Position',pos)

errorbar(sigs',aveAmp,stdAmp,'-o','MarkerSize',symbSize,'MarkerFaceColor',...
    greys(9,:),'MarkerEdgeColor',greys(9,:),'Color',greys(9,:),'LineWidth',...
    lineWd,'CapSize',0)
xlabel('$\sigma$','Interpreter','latex','FontSize',labelSize)
ylabel('Amplitude','FontSize',labelSize)
ylim([0.5,0.6])
% set(gca,'FontSize',axisSize,'LineWidth',axisWd)
set(gca,'FontSize',axisSize,'LineWidth',axisWd,'XScale','log','XTick',10.^(-3:1:-1))

% prefix = [figPre, 'pkAmp_sig'];
% saveas(f_sig_amp,[sFolder,filesep,prefix,'.fig'])
% print('-depsc',[sFolder,filesep,prefix,'.eps'])


% **********************************************
% D vs sigma
% **********************************************
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
% ebh.YNegativeDelta = []; % only show upper half of the error bar
% plot(sigs',y_pred,'Color',PuRd(7,:),'LineWidth',lineWd)

% exact theory, revised on 8/22/2021
x_sig = 10.^(-4:0.05:-1)';
plot(x_sig,(lr^2 + 16*lr*x_sig.^2)/2,'Color',PuRd(7,:),'LineWidth',lineWd)
hold off
box on
lg = legend('simulation','theory','Location','northwest');
% lg = legend('simulation',['$D \propto \eta^{',num2str(round(bs(2)*100)/100),'}$'],'Location','northwest');
set(lg,'Interpreter','latex','FontSize',16)
xlabel('$\sigma$','Interpreter','latex','FontSize',labelSize)
ylabel('$D$','Interpreter','latex','FontSize',labelSize)
set(gca,'FontSize',axisSize,'LineWidth',axisWd,'XScale','log','YScale','log',...
    'YTick',10.^(-5:1:-3),'XTick',10.^(-4:1:-1),'YAxisLocation','right')

% prefix = [figPre, 'pkAmp_sigma'];
% saveas(f_sig_D,[sFolder,filesep,prefix,'.fig'])
% print('-depsc',[sFolder,filesep,prefix,'.eps'])
% 

%% No explicit noise

% dFileN= fullfile('./data','pc1D_ring_Np_alp0_Np1_std0_lr0.05_bs1.mat');  % new simulation
dFileN= fullfile('../data_in_paper/','pc1D_ring_Np_alp0_Np1_std0_lr0.05_bs1_0217.mat');  % new simulation

load(dFileN)

aveDs_sample = nan(length(Nps),2);  % store average and standard deviation
for i = 1:length(Nps)
    if Nps(i) < 50
        temp = cat(2,allDs{i,:});
        aveDs_sample(i,:) = [nanmean(temp(:)),nanstd(temp(:))];
    else
        aveDs_sample(i,:) = [nanmean(allDs{i,1}),nanstd(allDs{i,1})];
    end
end

% heatmap of the diffusion for noise equals 0 case
sampleNoiseFig = figure;
set(sampleNoiseFig,'color','w','Units','inches','Position',[0,0,3.2,0.4])
imagesc(log10(aveDs_sample(:,1)'));
% colorbar
set(gca,'XTickLabel','','YTickLabel','')
%}
%% phase diagram
% x axis is the density of RF, y axis is the noise amplitude

dFolder = '../data_in_paper/pcRing_Ndp_0224';
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


% *************************************************
% heatmap of the summary diffusion constants and N
% *************************************************
fig_heat = figure;
set(fig_heat,'color','w','Units','inches','Position',pos)

imagesc(flip(log10(aveDs(Ix,1:9))))
colorbar
xlabel('$\log_2(N)$','Interpreter','latex','FontSize',20)
ylabel('$\log_{10}(\sigma)$','Interpreter','latex','FontSize',20)
% set(gca,'XTick',[1,3,6,9],'xticklabel',[0,2,5,8],'YTick',1:3:10,'yticklabel',[-4,-3.1,-2.2,-1.3])
% set(gca,'YTick',1:3:10,'yticklabel',[-4,-3.1,-2.2,-1.3],'FontSize',16)
set(gca,'YTick',[1,4,7,11],'yticklabel',flip([-4,-3.1,-2.2,-1]),'FontSize',16)

% title('$D,\alpha = 0$','Interpreter','latex')

% prefix = [figPre, 'ring_D_N_heatmap_0216'];
% saveas(fig_heat,[sFolder,filesep,prefix,'.fig'])
% print('-depsc',[sFolder,filesep,prefix,'.eps'])

 
% *************************************************
% heatmap, combine the case where explict noise is zero
% *************************************************
fig_heat = figure;
set(fig_heat,'color','w','Units','inches','Position',pos)

mergeM = [flip(log10(aveDs(Ix,1:10)));log10(aveDs_sample(1:10,1)')];
imagesc(mergeM)
colorbar
xlabel('$\log_2(N)$','Interpreter','latex','FontSize',20)
ylabel('$\log_{10}(\sigma)$','Interpreter','latex','FontSize',20)
% set(gca,'YTick',1:3:10,'yticklabel',[-4,-3.1,-2.2,-1.3],'FontSize',16)
set(gca,'YTick',[1,4,7,11],'yticklabel',flip([-4,-3.1,-2.2,-1]),'FontSize',16,...
    'XTick',1:2:10,'XTickLabel',0:2:8)

% prefix = [figPre, 'ring_D_N_heatmap_combined'];
% saveas(fig_heat,[sFolder,filesep,prefix,'.fig'])
% print('-depsc',[sFolder,filesep,prefix,'.eps'])




% example trace for small N and large N
figure
hold on
errorbar(sigs(Ix),aveDs(Ix,end),stdDs(Ix,end)','o-','MarkerSize',symbSize,'MarkerFaceColor',...
    greys(9,:),'MarkerEdgeColor',greys(9,:),'Color',greys(9,:),'LineWidth',...
    lineWd,'CapSize',0)
errorbar(sigs(Ix),aveDs(Ix,2),stdDs(Ix,2)','o-','MarkerSize',symbSize,'MarkerFaceColor',...
    greys(6,:),'MarkerEdgeColor',greys(6,:),'Color',greys(6,:),'LineWidth',...
    lineWd,'CapSize',0)
box on
legend('N = 256','N=4','Location','northwest')
xlabel('$\sigma$','Interpreter','latex','FontSize',labelSize)
ylabel('$D$','Interpreter','latex','FontSize',labelSize)
set(gca,'FontSize',axisSize,'LineWidth',axisWd,'XScale','log','YScale','log',...
    'YTick',10.^(-5:1:-1),'XTick',10.^(-4:1:-1))

% ************************************
% two examples
% ************************************
f_N_D_examp= figure;
hold on
set(f_N_D_examp,'color','w','Units','inches','Position',pos)

% errorbar(2.^(0:1:9)',aveDs(Ix(1),:)',stdDs(Ix(1),:)','o-','MarkerSize',symbSize,'MarkerFaceColor',...
%     greys(9,:),'MarkerEdgeColor',greys(9,:),'Color',greys(9,:),'LineWidth',...
%     lineWd,'CapSize',0)
errorbar(2.^(0:1:9)',aveDs_sample(:,1),aveDs_sample(:,2),'o-','MarkerSize',symbSize,'MarkerFaceColor',...
    greys(9,:),'MarkerEdgeColor',greys(9,:),'Color',greys(9,:),'LineWidth',...
    lineWd,'CapSize',0)
% errorbar(2.^(0:1:7)',aveDs(Ix(end-1),1:8)',stdDs(Ix(end-1),1:8)','o-','MarkerSize',symbSize,'MarkerFaceColor',...
%     blues(9,:),'MarkerEdgeColor',blues(9,:),'Color',blues(9,:),'LineWidth',...
%     lineWd,'CapSize',0)
errorbar(2.^(0:1:9)',aveDs(Ix(end-1),1:10)',stdDs(Ix(end-1),1:10)','o-','MarkerSize',symbSize,'MarkerFaceColor',...
    blues(9,:),'MarkerEdgeColor',blues(9,:),'Color',blues(9,:),'LineWidth',...
    lineWd,'CapSize',0)
hold off
box on
lg = legend('$\sigma = 0$','$\sigma = 0.05$','Location','northwest');
% lg = legend('$\sigma = 10^{-4}$','$\sigma = 0.1$','Location','northwest');
set(lg,'Interpreter','latex','FontSize',16)
xlabel('$N$','Interpreter','latex','FontSize',labelSize)
ylabel('$D$','Interpreter','latex','FontSize',labelSize)
ylim([1e-4,1e-1])
% ylim([1e-4,2e-2])
set(gca,'FontSize',axisSize,'LineWidth',axisWd,'XScale','log','YScale','log',...
    'YTick',10.^(-5:1:-1),'XTick',10.^(0:1:2))

% prefix = [figPre, 'ring_D_N_sample_Noise'];
% saveas(f_N_D_examp,[sFolder,filesep,prefix,'.fig'])
% print('-depsc',[sFolder,filesep,prefix,'.eps'])


% *************************************************
% Optimal N versus the noise standard deviation
% *************************************************
sortedDs = aveDs(Ix,:);
[~,I] = sort(sortedDs,2);


opt_N_sig= figure;
hold on
set(opt_N_sig,'color','w','Units','inches','Position',pos)
box on
plot(sigs(Ix),2.^(I(:,1)-1),'o-','MarkerSize',symbSize,'LineWidth',lineWd)
xlabel('$\sigma$','Interpreter','latex','FontSize',labelSize)
ylabel('Optimal $N$','Interpreter','latex','FontSize',labelSize)
set(gca,'FontSize',axisSize,'LineWidth',axisWd,'XScale','log','YScale','linear',...
    'XTick',10.^(-4:1:-1))



% *************************************************
% heatmap of the maximum mean square displacement
% *************************************************
figure
imagesc(allMaxMSD(Ix,:,1))
colorbar
title('Maximum MSD')
xlabel('$\log_2(N)$','Interpreter','latex')
ylabel('$\log_{10}(\sigma)$','Interpreter','latex')
set(gca,'YTick',1:3:10,'yticklabel',[-4,-3.1,-2.2,-1.3])
% set(gca,'XTick',[1,3,6,9],'xticklabel',[0,2,5,8],'YTick',1:3:10,'yticklabel',[-4,-3.1,-2.2,-1.3])


% *************************************************
% heatmap of the width of RF
% *************************************************
figure
imagesc(allWidthRF(Ix,:,1),[0.4,0.5])
colorbar
title('RF Width')
xlabel('$\log_2(N)$','Interpreter','latex')
ylabel('$\log_{10}(\sigma)$','Interpreter','latex')
% set(gca,'XTick',[1,3,6,9],'xticklabel',[0,2,5,8],'YTick',1:3:10,'yticklabel',[-4,-3.1,-2.2,-1.3])
set(gca,'YTick',1:3:10,'yticklabel',[-4,-3.1,-2.2,-1.3])


% *************************************************
% heatmap of active fraction
% *************************************************
figure
imagesc(allFracActi(Ix,:,1))
colorbar
title('Active fraction')
xlabel('$\log_2(N)$','Interpreter','latex')
ylabel('$\log_{10}(\sigma)$','Interpreter','latex')
% set(gca,'XTick',[1,3,6,9],'xticklabel',[0,2,5,8],'YTick',1:3:10,'yticklabel',[-4,-3.1,-2.2,-1.3])
set(gca,'YTick',1:3:10,'yticklabel',[-4,-3.1,-2.2,-1.3])

% *************************************************
% example two lines, with two different noise
% *************************************************
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

% prefix = [figPre, 'ring_D_N_actiFrac'];
% saveas(actiFrac_fig,[sFolder,filesep,prefix,'.fig'])
% print('-depsc',[sFolder,filesep,prefix,'.eps'])

%% Batch size effect on the sampling noise
% show how the optimal N shift due to the batch size
dFolder = '../data_in_paper/pcRing_bs_0205';
allFile = dir([dFolder,filesep,'*.mat']);
files = {allFile.name}';

% figPre = 'pcRing_phase';

s1 = '(?<= *_bs)[\d]+(?=\.mat)';

numNs = 16;    % could be 9 or 10, depending on the dataset
aveDs = nan(length(files),numNs);   % store all the mean Ds
stdDs = nan(length(files),numNs);   % store all the std of Ds
allMaxMSD = nan(length(files),numNs, 2);   % mean and standard
allWidthRF = nan(length(files),numNs, 2);   % mean and standard
allFracActi = nan(length(files),numNs, 2);   % mean and standard
bs = nan(length(files),1);    % store all the standard deviation


for i0 = 1:length(files)
    bs(i0) = str2num(char(regexp(files{i0},s1,'match')));
    raw = load(char(fullfile(dFolder,filesep,files{i0})));
    
    aveDs_noise = nan(numNs,2);  % store average and standard deviation
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


% order the bactch size
[~,Ix] = sort(bs,'ascend');


% *********************************************************
% heatmap of the summary diffusion constants and bach size
% *********************************************************
orderedDs = aveDs(Ix,:);
orderedStdDs = stdDs(Ix,:);

figure
imagesc(log10(aveDs(Ix,:)))
colorbar
% xlabel('$\log_2(N)$','Interpreter','latex')
% ylabel('$\log_{10}(\sigma)$','Interpreter','latex')
xlabel('$N$','Interpreter','latex')
ylabel('$\log_2(S)$','Interpreter','latex')
% set(gca,'XTick',[1,3,6,9],'xticklabel',[0,2,5,8],'YTick',1:3:10,'yticklabel',[-4,-3.1,-2.2,-1.3])
title('$D,\alpha = 0$','Interpreter','latex')

% prefix = [figPre, 'ring_D_N_heatmap'];
% saveas(gcf,[sFolder,filesep,prefix,'.fig'])
% print('-depsc',[sFolder,filesep,prefix,'.eps'])


% ************************************************
% Plot example trace
% ************************************************
pInx = [1,5,9];

f_bs = figure;
set(f_bs,'color','w','Units','inches','Position',pos)
hold on
for i = 1:length(pInx)
errorbar(raw.Nps,orderedDs(pInx(i),:)',orderedStdDs(pInx(i),:)','o-','MarkerSize',...
    symbSize,'MarkerFaceColor',greys(2+3*i,:),'MarkerEdgeColor',greys(2+3*i,:),...
    'Color',greys(2+3*i,:),'LineWidth',lineWd,'CapSize',0)
end
hold off
legend('s = 1','s = 16','s = 258')
box on
xlabel('$N$','Interpreter','latex','FontSize',20)
ylabel('$D$','Interpreter','latex','FontSize',20)
set(gca,'YScale','log','XScale','log','FontSize',16)


% *************************************************
% heatmap of the maximum mean square displacement
% *************************************************
figure
imagesc(log(allMaxMSD(Ix,:,1)))
colorbar
title('Maximum MSD','FontSize',20)
xlabel('$N$','Interpreter','latex')
ylabel('$\log_2(S)$','Interpreter','latex')
% set(gca,'XTick',[1,3,6,9],'xticklabel',[0,2,5,8],'YTick',1:3:10,'yticklabel',[-4,-3.1,-2.2,-1.3])


% example traces
msd_bs = figure;
set(msd_bs,'color','w','Units','inches','Position',pos)
hold on

for i = 1:length(pInx)
errorbar(raw.Nps,allMaxMSD(Ix(pInx(i)),:,1)',allMaxMSD(Ix(pInx(i)),:,2)','o-','MarkerSize',...
    symbSize,'MarkerFaceColor',greys(2+3*i,:),'MarkerEdgeColor',greys(2+3*i,:),...
    'Color',greys(2+3*i,:),'LineWidth',lineWd,'CapSize',0)
end
hold off
legend('s = 1','s = 16','s = 512')
box on
xlabel('$N$','Interpreter','latex','FontSize',20)
ylabel('$MSD$','Interpreter','latex','FontSize',20)
set(gca,'YScale','log','XScale','log','FontSize',16)




% *************************************************
% heatmap of the width of RF
% *************************************************
figure
imagesc(allWidthRF(Ix,:,1),[0.4,0.5])
colorbar
title('RF Width')
xlabel('$\log_2(N)$','Interpreter','latex')
% ylabel('$\log_{10}(\sigma)$','Interpreter','latex')
ylabel('$\log_2(S)$','Interpreter','latex')
set(gca,'XTick',[1,3,6,9],'xticklabel',[0,2,5,8])
