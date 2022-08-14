% thie program replot some of the pannels in figure 2 of Driscol et al 207
% data are extracted using adobe illustrator
% 

close all
clear

%% 
defaultGraphicsSetttings
greys = brewermap(11,'Greys');
blues = brewermap(11,'blues');
thisBlue = [52,153,204]/256;
thisRed = [218,28,92]/256;
thisBlack = [0,0,0];

saveFolder = '../figures';
figWidth = 3.2;
figHeight = 2.8;

%% fraction of active cells
days = [1,10,20];
refLen1 = 97.85;  % reference of 0.3
fracActi = [113.9, 161.1, 182.4;95.2,137.7,150.4;60.7,95.67,101.5]/refLen1*0.3;
stdFrac = [13.1,30.8/2,30.8/2;26.8/2,15.8,22.8;17.1/2,23.8/2,47.7/2]/refLen1*0.3;

fracFig = figure;
pos(3)=3.4; 
pos(4)=2.5;
set(fracFig,'color','w','Units','inches','Position',pos)

hold on
for i= 1:3
    errorbar(days',fracActi(i,:)',stdFrac(i,:)','o-','MarkerSize',8,'Color',blues(2+3*i,:),...
    'MarkerEdgeColor',blues(1+3*i,:),'MarkerFaceColor',blues(1+3*i,:),'CapSize',0,...
    'LineWidth',1.5)
end

box on
hold off
ylim([0,0.7])
xlim([-3,22])
xlabel('$\Delta$ days','Interpreter','latex')
ylabel({'Fraction of','peak moved'},'FontSize',20)
% ylabel('Fraction of peak moved','FontSize',20)
set(gca,'FontSize',16,'XTick',[1,10,20],'LineWidth',1)

prefix = 'ppc_Drico_peakMoved';
saveas(gcf,[saveFolder,filesep,prefix,'.fig'])
print('-depsc',[saveFolder,filesep,prefix,'.eps'])


%% active fraction
days = [1,10,20];
refLen1 = 43.1;  % reference of 0.3
fracActi = [47.45, 43.5, 44.8]/refLen1*0.3;
stdFrac = [22.2/2,9.45/2,8.5/2]/refLen1*0.3;

fracFig = figure;
pos(3)=3.3; 
pos(4)=2.5;
set(fracFig,'color','w','Units','inches','Position',pos)

errorbar(days',fracActi',stdFrac','o-','MarkerSize',8,'Color',greys(11,:),...
    'MarkerEdgeColor',greys(11,:),'MarkerFaceColor',greys(11,:),'CapSize',0,...
    'LineWidth',1.5)

box on
hold off
ylim([0,0.7])
xlim([-3,22])
xlabel('$\Delta$ days','Interpreter','latex')
ylabel({'Fraction of cells','with peaks'},'FontSize',20)
% ylabel('Fraction of peak moved','FontSize',20)
set(gca,'FontSize',16,'XTick',[1,10,20],'LineWidth',1)

% prefix = 'ppc_Drico_fraction_active';
% saveas(gcf,[saveFolder,filesep,prefix,'.fig'])
% print('-depsc',[saveFolder,filesep,prefix,'.eps'])


%% Fraction of neurons with loss and gain peak responses
% loss 
ylow = 1.489;
yhigh = 0.4888;
yrange = 0.5;
ys = [1.045;0.750;0.626];  % average values
ystd = [1.045;0.82;0.728];

yval = abs(ys - ylow)/(ylow - yhigh)*yrange;
stdVal = abs(ystd - ys)/(ylow - yhigh)*yrange;

% gain
gYlow  = 1.488;
gYhigh = 0.555;
gYrange = 0.15;
gYs = [0.964;0.686;0.758];
gYstd = [0.964;0.815;0.911];
gYval = abs(gYs - gYlow)/(gYlow - gYhigh)*gYrange;
gstdVal = abs(gYstd - gYs)/(gYlow - gYhigh)*gYrange;


% plot the figure 
lossGainFig = figure;
pos(3)=3.1; 
pos(4)=2.5;
set(lossGainFig,'color','w','Units','inches','Position',pos)
hold on
errorbar(days',yval,stdVal,'o-','MarkerSize',8,'Color',thisRed,...
    'MarkerEdgeColor',thisRed,'MarkerFaceColor',thisRed,'CapSize',0,...
    'LineWidth',1.5)
errorbar(days',gYval,gstdVal,'o-','MarkerSize',8,'Color',thisBlue,...
    'MarkerEdgeColor',thisBlue,'MarkerFaceColor',thisBlue,'CapSize',0,...
    'LineWidth',1.5)

box on
hold off
ylim([0,0.7])
xlim([-3,22])
xlabel('$\Delta$ Days','Interpreter','latex')
ylabel({'Fraction'},'FontSize',20)
% ylabel('Fraction of peak moved','FontSize',20)
set(gca,'FontSize',16,'XTick',[1,10,20],'LineWidth',1)

% prefix = 'ppc_Drico_fraction_loss_gain';
% saveas(gcf,[saveFolder,filesep,prefix,'.fig'])
% print('-depsc',[saveFolder,filesep,prefix,'.eps'])