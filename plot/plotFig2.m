% plot Figure 2 PSP 
% data is generated from running "drift_PSP.m"

% load the data
dFile = '../data/PSPonline/PSP_eta0.05_sig0.01_0824.mat';  % eta = 0.05
% dFile = '../data/PSPonline/PSP_eta0.1_sig0.01_0824.mat';  % eta = 0.1

load(dFile)
sFolder = '../figures';
%% Graphics settings
% divergent colors, used in heatmap
nc = 256;   % number of colors
spectralMap = brewermap(nc,'Spectral');
PRGnlMap = brewermap(nc,'PRGn');
RdBuMap = flip(brewermap(nc,'RdBu'),1);

blues = brewermap(11,'Blues');
BuGns = brewermap(11,'BuGn');
greens = brewermap(11,'Greens');
greys = brewermap(11,'Greys');
PuRd = brewermap(11,'PuRd');
YlOrRd = brewermap(11,'YlOrRd');
oranges = brewermap(11,'Oranges');

set1 = brewermap(11,'Set1');

%% ======================== Figure 2 in the main text ====================
plot_offline_flag = false;  % whether or not plot offline data

fig2 = figure; 
set(gcf,'color','w','Units','inches')
pos(3)=16.5;  %17.4; %8.5;%pos(3)*1.5;
pos(4)=8;%pos(4)*1.5;
set(gcf,'Position',pos)
labelFontSize = 24;
gcaFontSize = 20;


% example trajectories
aAxes = axes('position',[.07  .63  0.2  0.35]); hold on
annotation('textbox', [.005 .98 .03 .03],...
    'String', 'A','BackgroundColor','none','Color','k',...
    'LineStyle','none','fontsize',labelFontSize,'HorizontalAlignment','Center');

% example trajectories
bAxesi = axes('position',[.33  .63  .16  .3]); hold on
bAxesii = axes('position',[.5  .63  .16  .3]); hold on
annotation('textbox', [.29 .98 .03 .03],...
    'String', 'B','BackgroundColor','none','Color','k',...
    'LineStyle','none','fontsize',labelFontSize,'HorizontalAlignment','Center');


% Ensemble data clouds
cAxes = axes('position',[.76  .63  .2  .35]); hold on
annotation('textbox', [.72 .98 .03 .03],...
    'String', 'C','BackgroundColor','none','Color','k',...
    'LineStyle','none','fontsize',labelFontSize,'HorizontalAlignment','Center');

% Example trajectory of one neuron
dAxes = axes('position',[.07  .12  0.2  0.35]); hold on
annotation('textbox', [.005 .47 .03 .03],...
    'String', 'D','BackgroundColor','none','Color','k',...
    'LineStyle','none','fontsize',labelFontSize,'HorizontalAlignment','Center');

% D vs noise amplitude
eAxes = axes('position',[.4  .12  .22  .35]);
annotation('textbox', [.33 .47 .03 .03],...
    'String', 'E','BackgroundColor','none','Color','k',...
    'LineStyle','none','fontsize',labelFontSize,'HorizontalAlignment','Center');

% D vs eigen spectrum
fAxes = axes('position',[.7  .12  .22  .35]); 
annotation('textbox', [.63 .47 .03 .03],...
    'String', 'F','BackgroundColor','none','Color','k',...
    'LineStyle','none','fontsize',labelFontSize,'HorizontalAlignment','Center');

% **************************************************
% plot the example trajectory
%**************************************************
% selYInx = randperm(num_sel,1);
selYInx = 57;   % for eta = 0.05 
ys = Yt(:,:,selYInx);
colorsSel = [YlOrRd(6,:);PuRd(8,:);blues(8,:)];

pointGap = 20;
xs = (1:pointGap:time_points)'*step;

axes(aAxes)
for i = 1:output_dim
    plot(xs,ys(i,1:pointGap:time_points)','Color',colorsSel(i,:),'LineWidth',2)
    hold on
end
hold off
box on
bLgh = legend('y_1','y_2','y_3','Location','northwest');
legend boxoff
set(bLgh,'FontSize',14)
xlabel('$t$','Interpreter','latex','FontSize',labelFontSize)
ylabel('$y_i(t)$','Interpreter','latex','FontSize',labelFontSize)
set(aAxes,'LineWidth',1,'FontSize',gcaFontSize)

% **************************************************
% plot the similarity marix at two time points
% **************************************************
smSelect = randperm(num_sel,20);
% smSelect = sm_example;
time1 = 2e3;
time2 = 7e3;
Y1sel = squeeze(Yt(:,time1,smSelect));
Y2sel = squeeze(Yt(:,time2,smSelect));

% order the idex by the cluster
tree = linkage(Y1sel','average');
D = pdist(Y1sel');
leafOrder = optimalleaforder(tree,D);

axes(bAxesi)
imagesc(Y1sel(:,leafOrder)'*Y1sel(:,leafOrder),[-6,6])
colormap(bAxesi,RdBuMap)
box on
set(bAxesi,'XTick',[1,10,20],'YTick',[1,10,20],'YLim',[0.5,20.5],'XLim',...
    [0.5,20.5],'FontSize',gcaFontSize)
xlabel('Stimuli','FontSize',labelFontSize)
ylabel('Stimuli','FontSize',labelFontSize)
title(bAxesi,'$t = 1$','Interpreter','latex','FontSize',labelFontSize)

axes(bAxesii)
imagesc(Y2sel(:,leafOrder)'*Y2sel(:,leafOrder),[-6,6])
colormap(bAxesii,RdBuMap)
box on
set(bAxesii,'XTick',[1,10,20],'YTick','','YLim',[0.5,20.5],'XLim',[0.5,20.5],...
    'FontSize',gcaFontSize)
c = colorbar;
c.Position = [0.67,0.63,0.01,0.3];
xlabel('Stimuli','FontSize',labelFontSize)
title(bAxesii,'$t = 5\times 10^4$','Interpreter','latex','FontSize',labelFontSize)

% **************************************************
% ensemble data cloud
% **************************************************
% ensem_sel = [1, 6];   % when eta = 0.05
ensem_sel = [2, 7]; % when eta = 0.1
axes(cAxes)
for i = 1:length(ensem_sel)
    scatterHd = plot3(Y_ensemble(1,:,ensem_sel(i)),Y_ensemble(2,:,ensem_sel(i)),...
        Y_ensemble(3,:,ensem_sel(i)),'.','MarkerSize',6);
    hold on
    scatterHd.MarkerFaceColor(4) = 0.2;
    scatterHd.MarkerEdgeColor(4) = 0.2;
    grid on
end
hold off
% lg = legend('$t = 0$','$t= 5\times 10^4$','interpreter','latex')
xlabel('$y_1$','Interpreter','latex','FontSize',labelFontSize)
ylabel('$y_2$','Interpreter','latex','FontSize',labelFontSize)
zlabel('$y_3$','Interpreter','latex','FontSize',labelFontSize)
set(gca,'FontSize',gcaFontSize,'XLim',[-4,4],'YLim',[-4,4],'ZLim',[-3,3])

% ************************************************************
% Trajectory of an example output
% ************************************************************
% exampleSel = randperm(num_sel,1);  % randomly select a 
exampleSel = 182; % 150 for eta = 0.05, and 135 for eta = 0.1
Yexample = Yt(:,:,exampleSel);
interSep = 5;   % only select data every 10 time points, to eliminate clutter
dotColors = flip(brewermap(size(Yexample,2)/interSep,'Spectral'));

% Also plot a sphere to guid visualization
ridus = sqrt(mean(sum(Yexample.^2,1)));
gridNum = 30;
u = linspace(0, 2 * pi, gridNum);
v = linspace(0, pi, gridNum);

axes(dAxes)
sfh = surf(ridus * cos(u)' * sin(v), ridus * sin(u)' * sin(v), ...
    ridus * ones(size(u, 2), 1) * cos(v), 'FaceColor', 'w', 'EdgeColor', [.9 .9, .9]);
sfh.FaceAlpha = 0.3000;  % for transparency
hold on
for i = 1:(size(Yexample,2)/interSep)
    plot3(Yexample(1,i*interSep),Yexample(2,i*interSep),Yexample(3,i*interSep),...
        '.','MarkerSize',8,'Color',dotColors(i,:))
end
% grid on
hold off
colormap(dAxes,dotColors)
c2 = colorbar;
c2.Position = [0.25,0.2,0.01,0.2];

% enhance visualization
radius_new = ridus;   % this should be modified based on the effect
set(dAxes,'XLim',[-radius_new,radius_new],'YLim',[-radius_new,radius_new],...
    'ZLim',[-radius_new,radius_new])

view([45,30])
xlabel('$y_1$','Interpreter','latex','FontSize',labelFontSize)
ylabel('$y_2$','Interpreter','latex','FontSize',labelFontSize)
zlabel('$y_3$','Interpreter','latex','FontSize',labelFontSize)
set(dAxes,'FontSize',gcaFontSize,'LineWidth',1.5)

% **************************************************
%  D vs noise amplitude
% **************************************************
% first, load the data
noiseData_offline = load('../data/PSP_offline/noiseAmp_offline_08242022.mat');          % offline
noiseData_online = load('../data/PSPonline/noiseAmp_online_0824_01.mat'); 

% offline data
out_offline_amp = PSP_noise_data_aggregate(noiseData_offline,'refit');

% online data
out_online_amp = PSP_noise_data_aggregate(noiseData_online,'refit');

selInx = 1:2:out_offline_amp.num_std;  % select part of the data to fit

axes(eAxes)
eh2 = errorbar(out_online_amp.noiseAmpl(selInx)',out_online_amp.aveD_noise(selInx)',...
    out_online_amp.stdD_noise(selInx)','o','MarkerSize',8,'MarkerFaceColor',...
    greys(7,:),'Color',greys(7,:),'LineWidth',1.5,'CapSize',0);
eh2.YNegativeDelta = []; % only show upper half of the error bar

hold on
if plot_offline_flag
    eh1 = errorbar(out_offline_amp.noiseAmpl(selInx)',out_offline_amp.aveD_noise(selInx)',...
        out_offline_amp.stdD_noise(selInx)','o','MarkerSize',8,'MarkerFaceColor',...
        blues(7,:),'Color',blues(7,:),'LineWidth',1.5,'CapSize',0);
    eh1.YNegativeDelta = []; % only show upper half of the error bar
end

plot(eAxes,out_offline_amp.noiseAmpl(selInx)',out_offline_amp.D_pred(selInx),'LineWidth',2,'Color',PuRd(7,:))
hold off

if plot_offline_flag
    lg = legend('Oflline','Online','theory','Location','northwest');
    set(lg,'FontSize',gcaFontSize)
else
    lg = legend('simulation','theory','Location','northwest');
    set(lg,'FontSize',gcaFontSize)
end
set(lg,'Interpreter','Latex')
legend boxoff
xlabel('Noise amplitude $(\sigma^2)$','Interpreter','latex','FontSize',labelFontSize)
ylabel('$D_{\varphi}$','Interpreter','latex','FontSize',labelFontSize)
ylim([5e-9,2e-4])
% ylim([1e-6,1e-4])
set(gca,'LineWidth',1.5,'FontSize',gcaFontSize,'XScale','log','YScale','log',...
    'YTick',10.^(-8:2:-4))


% ***************************************************
% D vs eigenspectrum
% ***************************************************
% spectrumData_offline = load('../data/eigenSpectr0821.mat');  % offline
spectrumData_offline = load('../data/PSP_offline/eigenSpectr_offline_0825_005.mat');  % offline
% spectrumData_online = load('../data/eigenSpectr_08232022_1.mat'); % online
spectrumData_online = load('../data/PSPonline/eigenSpectr_online_0825_005.mat'); % online


out_offline = PSP_spectrum_data_aggregate(spectrumData_offline,'refit');
out_online = PSP_spectrum_data_aggregate(spectrumData_online,'refit');


axes(fAxes)
eh2 = errorbar(out_online.centers',out_online.aveSpDs(:,1),out_online.aveSpDs(:,2),...
    'o','MarkerSize',8,'MarkerFaceColor',...
    greys(7,:),'Color',greys(7,:),'LineWidth',1,'CapSize',0);
eh2.YNegativeDelta = []; % only show upper half of the error bar

hold on
if plot_offline_flag
    eh1 = errorbar(out_offline.centers',out_offline.aveSpDs(:,1),out_offline.aveSpDs(:,2),...
    'o','MarkerSize',8,'MarkerFaceColor',blues(7,:),'Color',blues(7,:),'LineWidth',1,'CapSize',0);
    eh1.YNegativeDelta = []; % only show upper half of the error bar
end

% theory
plot(out_offline.centers',out_offline.theoPre*out_offline.centers','LineWidth',2,'Color',PuRd(7,:))
if plot_offline_flag
    lg = legend('Offline','Online','theory','Location','northwest');
    set(lg,'FontSize',gcaFontSize)
else
    lg = legend('simulation','theory','Location','northwest');
    set(lg,'FontSize',gcaFontSize)
end
hold off
xlabel('$\sum_{i=1}^{k}1/\lambda_i^2$','Interpreter','latex','FontSize',labelFontSize)
ylabel('$D_{\varphi}$','Interpreter','latex','FontSize',labelFontSize)
set(fAxes,'LineWidth',1.5,'FontSize',gcaFontSize,'XScale','log','YScale','log',...
    'Ylim',[1e-7,2e-4],'YTick',10.^(-7:1:-4),'XTick',10.^([-1,0,1,2]))

% prefix = ['psp_fig2_',num2str(learnRate),'_no_offline',date,];
% saveas(fig2,[sFolder,filesep,prefix,'.fig'])
% print('-depsc',[sFolder,filesep,prefix,'.eps'])
% print('-painters','-dpdf',[saveFolder,filesep,prefix,'.pdf'])

%% Fig S2
figS2 = figure; 
set(figS2,'color','w','Units','inches')
pos(3)=10;  %17.4; %8.5;%pos(3)*1.5;
pos(4)=8;%pos(4)*1.5;
set(gcf,'Position',pos)


% PSP error initial stage
aAxes = axes('position',[.14  .63  0.35  0.35]); hold on
annotation('textbox', [.005 .98 .03 .03],...
    'String', 'A','BackgroundColor','none','Color','k',...
    'LineStyle','none','fontsize',labelFontSize,'HorizontalAlignment','Center');

% PSP error stationary state
bAxes = axes('position',[.63  .63  .35  .35]);
annotation('textbox', [.52 .98 .03 .03],...
    'String', 'B','BackgroundColor','none','Color','k',...
    'LineStyle','none','fontsize',labelFontSize,'HorizontalAlignment','Center');

% change of similarity matrix
cAxes = axes('position',[.14  .12  0.35  0.38]); hold on
annotation('textbox', [.005 .49 .03 .03],...
    'String', 'C','BackgroundColor','none','Color','k',...
    'LineStyle','none','fontsize',labelFontSize,'HorizontalAlignment','Center');

% auto-correlation function
dAxes = axes('position',[.63  .12  .35  .38]);
annotation('textbox', [.52 .49 .03 .03],...
    'String', 'D','BackgroundColor','none','Color','k',...
    'LineStyle','none','fontsize',labelFontSize,'HorizontalAlignment','Center');

% learning curve at initial stage
axes(aAxes)
plot(((1:length(initial_pspErr))*step)',initial_pspErr,'LineWidth',3)
xlabel('$t$','Interpreter','latex','FontSize',labelFontSize)
ylabel('PSP error','FontSize',labelFontSize)
set(gca,'FontSize',gcaFontSize,'LineWidth',1)
xlim([1 1e3])
box on

% plot PSP error
pointGap = 20;
xs = (1:pointGap:time_points)'*step;
axes(bAxes)
plot(xs,pspErr(1:pointGap:time_points),'Color',greys(8,:),'LineWidth',1)
box on
xlabel('$t$','Interpreter','latex','FontSize',labelFontSize)
ylabel('PSP error','FontSize',labelFontSize)
set(bAxes,'LineWidth',1,'FontSize',gcaFontSize,'Ylim',[0,1])

% change of similarity matrix norm, and F'F norm compared with identity
% matrix
axes(cAxes)
plot(xs,SMerror(1:pointGap:time_points),'Color',blues(9,:),'LineWidth',1)
box on
xlabel('$t$','Interpreter','latex','FontSize',labelFontSize)
ylabel('$||\rm{SM_{t} - SM_{0}}||_F/||SM_0||_F$','Interpreter','latex',...
    'FontSize',labelFontSize)
set(cAxes,'LineWidth',1,'FontSize',gcaFontSize,'YLim',[0,0.2])


% =================================================================
% plot example rotational means quare displacement and a fit of the
% diffusiotn constant
% =================================================================
fitRange = 1000;  % only select partial data to fit
aveRMSD = mean(rmsd,1);
logY = log(aveRMSD(1:fitRange)');
logX = [ones(fitRange,1),log((1:fitRange)'*step)];
b = logX\logY;  
Dphi = exp(b(1));  % diffusion constant
exponent = b(2); % factor
pInx = randperm(size(rmsd,1),20);

axes(dAxes)
plot((1:size(rmsd,2))'*step, rmsd(pInx,:)','Color',greys(5,:),'LineWidth',1.5)
hold on

% overlap fitted line
yFit = exp(logX*b);
plot((1:fitRange)'*step,yFit,'k--','LineWidth',2)
hold off
xlabel('$\Delta t$','Interpreter','latex','FontSize',labelFontSize)
ylabel('$\langle\varphi^2(\Delta t)\rangle$','Interpreter','latex','FontSize',labelFontSize)
set(gca,'FontSize',20,'LineWidth',1,'XScale','log','YScale','log',...
 'XTick',10.^(1:5),'YTick',10.^(-4:2:2))
set(gca,'FontSize',gcaFontSize,'LineWidth',1.5)


% set(gcf, 'Renderer', 'Painters');
% prefix = ['psp_S1_eta005_',date,];
% saveas(figS2,[sFolder,filesep,prefix,'.fig'])
% print('-depsc',[sFolder,filesep,prefix,'.eps'])
