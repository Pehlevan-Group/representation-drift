% Prepare the figures of the 2D place cell simulation
%
% the default dataset is './data/placeCell_1029_2.mat'
dFile = 'pc2D_new_0215.mat';
load(fullfile('../data/',dFile))

%% Graphics setting
% this part polish some of the figures and make them publication ready
% define all the colors
sFolder = '../figures';
figPre = 'pCell2D_0215_';

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
set1 = brewermap(11,'Set1');


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

%%
% ************************************************************
% peak position, using time point 1
% ************************************************************
temp = [ceil(pks(:,1)/param.ps),mod(pks(:,1),param.ps)]/param.ps;
pkVec = temp(~isnan(temp(:,1)),:);

% define figure size
f_pkPosi = figure;
set(f_pkPosi,'color','w','Units','inches','Position',pos)
% histogram(activeInter(:))
% xlim([0,100])
plot(pkVec(:,1),pkVec(:,2),'+','MarkerSize',3,'MarkerFaceColor',greys(9,:),...
    'MarkerEdgeColor',greys(9,:),'LineWidth',1.5)
xlabel('X position','FontSize',labelSize)
ylabel('Y position','FontSize',labelSize)
set(gca,'LineWidth',1,'FontSize',axisSize)

% prefix = [figPre, 'pkPosi'];
% saveas(f_pkPosi,[sFolder,filesep,prefix,'.fig'])
% print('-depsc',[sFolder,filesep,prefix,'.eps'])

%%
% ************************************************************
% peak amplitude of example neuron, Fig S4B
% ************************************************************
% inx = 10;  % select one neuron to plot, 10 for Wij noise only
inx = 150;
temp = pkAmp(inx,:);
temp(isnan(temp)) = 0;
times = (1:length(temp))*step;   % iterations

f_pkAmpTraj = figure;
set(f_pkAmpTraj,'color','w','Units','inches','Position',pos)

plot(times(1:1000)',temp(1:1000)','LineWidth',1.5,'Color',blues(9,:))
xlabel('Time','FontSize',labelSize)
ylabel('Peak Amplitude','FontSize',labelSize)
set(gca,'LineWidth',1,'FontSize',axisSize)

% prefix = [figPre, 'pkAmp'];
% saveas(f_pkAmpTraj,[sFolder,filesep,prefix,'.fig'])
% print('-depsc',[sFolder,filesep,prefix,'.eps'])


% ************************************************************
% Fraction of active neurons
% ************************************************************
f_acti = figure;
set(f_acti,'color','w','Units','inches','Position',pos)
actiFrac = mean(pks > 0,1);

plot(times',actiFrac','LineWidth',lineWd)
xlabel('Time','FontSize',labelSize)
ylabel('Active fraction','FontSize',labelSize)
ylim([0,1])
set(gca,'LineWidth',1,'FontSize',axisSize)

% prefix = [figPre, 'fracActive'];
% saveas(f_acti,[sFolder,filesep,prefix,'.fig'])
% print('-depsc',[sFolder,filesep,prefix,'.eps'])

%% trajectory of an example neuron, Fig 5C

% epInx = randperm(param.Np,1);  % randomly slect
epInx = 105;
epPosi = [floor(pks(epInx,:)/param.ps);mod(pks(epInx,:),param.ps)]+randn(2,size(pks,2))*0.1;

specColors = brewermap(size(epPosi,2), 'Spectral');

% colors indicate time
f_pkPosi = figure;
set(f_pkPosi,'color','w','Units','inches')
pos(3)=3.2;  
pos(4)=2.8;
set(f_pkPosi,'Position',pos)
hold on

for i=1:size(pks,2)
    plot(epPosi(1,i),epPosi(2,i),'o','MarkerSize',4,'MarkerEdgeColor',...
        specColors(i,:),'LineWidth',1.5)
end
hold off
box on
xlabel('x position','FontSize',16)
ylabel('y position','FontSize',16)
set(gca,'FontSize',16,'LineWidth',1)

% prefix = [figPre, 'pkPosi_example'];
% saveas(f_pkPosi,[sFolder,filesep,prefix,'.fig'])
% print('-depsc',[sFolder,filesep,prefix,'.eps'])

%%
% ************************************************************
% Representational similarity across time, Fig S4A
% ************************************************************
% for better viusalization, only use part of the data
% gray color maps
selInx = 1:10:1024;
Y1= Yt(:,:,1);
Y2 = Yt(:,:,400);
SM1 = Y1'*Y1;
SM2 = Y2'*Y2;
% SM1 = Y1(:,selInx)'*Y1(:,selInx);
% SM2 = Y2(:,selInx)'*Y2(:,selInx);

f_sm1 = figure;
set(f_sm1,'color','w','Units','inches','Position',pos)

imagesc(SM1,[0,25]);
colormap(viridis)
cb = colorbar;
title('$t = 1$','Interpreter','latex')
set(cb,'FontSize',12)
set(gca,'XTick','','YTick','','LineWidth',0.5)
% 
% prefix = [figPre, 'sm1'];
% saveas(f_sm1,[sFolder,filesep,prefix,'.fig'])
% print('-depsc',[sFolder,filesep,prefix,'.eps'])

f_sm2 = figure;
set(f_sm2,'color','w','Units','inches','Position',pos)

imagesc(SM2,[0,25]);
colormap(viridis)
cb = colorbar;
set(cb,'FontSize',12)
title('$t = 4000$','Interpreter','latex')
set(gca,'XTick','','YTick','','LineWidth',0.5)
% prefix = [figPre, 'sm2'];
% saveas(f_sm2,[sFolder,filesep,prefix,'.fig'])
% print('-depsc',[sFolder,filesep,prefix,'.eps'])

%% Estimate the diffusion constants, vs averaged peak, Fig S4G

msds = nan(floor(time_points/2),size(pks,1));
xcoord = floor(pks/param.ps);
ycoord = mod(pks,param.ps);
for i = 1:size(msds,1)
    
    if param.Np == 1
        t1 = abs(xcoord(i+1:end)' - xcoord(1:end-i)');
        t2 = abs(ycoord(i+1:end)' - ycoord(1:end-i)');
    else
        t1 = abs(xcoord(:,i+1:end,:) - xcoord(:,1:end-i,:));
        t2 = abs(ycoord(:,i+1:end,:) - ycoord(:,1:end-i,:));
    end
    
    dx = min(t1,param.ps - t1);
    dy = min(t2,param.ps - t2);
    msds(i,:) = nanmean(dx.^2 + dy.^2,2)';
end


% linear regression to get the diffusion constant of each neuron
Ds = PlaceCellhelper.fitLinearDiffusion(msds,step,'linear');

% ***********************************************************************
% D vs mean amplitude
% ***********************************************************************

f_ampDiff= figure;
set(f_ampDiff,'color','w','Units','inches','Position',pos)

plot(meanPks,Ds,'o','MarkerSize',symbSize,...
    'MarkerEdgeColor',greys(9,:),'LineWidth',lineWd)
xlabel('Average peak amplitude','FontSize',labelSize)
ylabel('$D$','Interpreter','latex','FontSize',labelSize)
% set(gca,'FontSize',axisSize,'LineWidth',axisWd,'YScale','log','YTick',10.^(-3:0))
set(gca,'FontSize',axisSize,'LineWidth',axisWd)

% prefix = [figPre, 'batch_diffu_Amp'];
% saveas(f_ampDiff,[sFolder,filesep,prefix,'.fig'])
% print('-depsc',[sFolder,filesep,prefix,'.eps'])


% ***********************************************************************
% Fraction of active time vs mean amplitude
% ***********************************************************************

f_ampActi= figure;
set(f_ampActi,'color','w','Units','inches','Position',pos)

plot(meanPks,tolActiTime,'o','MarkerSize',symbSize,...
    'MarkerEdgeColor',greys(9,:),'LineWidth',lineWd)
xlabel('Average peak amplitude','FontSize',labelSize)
ylabel('Fraction of active time','FontSize',labelSize)
set(gca,'FontSize',axisSize,'LineWidth',axisWd)

% prefix = [figPre, 'batch_actiTime_Amp'];
% saveas(f_ampActi,[sFolder,filesep,prefix,'.fig'])
% print('-depsc',[sFolder,filesep,prefix,'.eps'])

%% Histogram of active and silent intervals, Fig S4C

lg = legend('simulation','exponential fit');

xlabel('Silent interval','FontSize',16)
ylabel('Pdf','FontSize',16)
set(gca,'LineWidth',1,'FontSize',16)

% prefix = [figPre, 'batch_actiTime_hist'];
% saveas(f_silentInter,[sFolder,filesep,prefix,'.fig'])
% print('-depsc',[sFolder,filesep,prefix,'.eps'])