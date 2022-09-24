% This program extract the data from Driscoll 2017 figure 3F
% and plot the fraction of consistent, switched and gained tuning to
% compare with our model

close all

%% prepare the data
% the input is the image of a heatmap

fig = '../data_in_paper/Driscoll_fig3F.png';

I = imread(fig);
figure
imshow(I)
rows = 1:9:1287;
cols = 50 + (0:10)*53;


rdmap = redblue(256);
% convert to index
IND = rgb2ind(I,rdmap,'nodither');
mat = IND(rows,cols);



%% analysis
stdVal = std(double(mat(:)));
% thdLeft = 128 - stdVal;
% thdRight = 128 + stdVal;
margin = 32;
thdLeft = 128-margin;
thdRight = 128+margin;

%left tune
leftMat = mat < thdLeft;
figure
imagesc(leftMat)


% right tune
rightMat = mat > thdRight;
figure
imagesc(rightMat)


% estimate the stability of tuning
days =size(mat,2)-1;
stabMat = nan(days,3); % consistent, loss and gain

%% Another method
days =size(mat,2)-1;
stabMat = nan(days,3); % consistent, loss and gain
numNeuron = size(mat,1);
tuningInfo = nan(numNeuron,days+1);
tuningInfo(mat < thdLeft) = -1;  % left tuning
tuningInfo(mat > thdRight) = 1;    % right tuning

for dt = 1:days
    % consistent
    tempC = tuningInfo(:,1+dt:days+1) - tuningInfo(:,1:days+1-dt);
    refTuning = mean(~isnan(tuningInfo(:,1:days+1-dt)),1);
    stabMat(dt,1) = mean(mean(tempC==0,1)./refTuning);
    
    
    % loss of tuning
    tmp1 = ~isnan(tuningInfo(:,1:days+1-dt));
    tmp2 = isnan(tuningInfo(:,1+dt:days+1));
    stabMat(dt,2) = mean(mean(tmp1.*tmp2,1)./refTuning);
    
    % shifted
    stabMat(dt,3) = mean(mean(abs(tempC) > 0,1)./refTuning);
    
end

figure
plot(stabMat)
%% plot the figure
% figure settings
% blues = brewermap(11,'Blues');
% greys = brewermap(11,'Greys');
% PuRd = brewermap(11,'PuRd');
thisBlue = [52,153,204]/256;
thisRed = [218,28,92]/256;
thisBlack = [0,0,0];
sFolder = '../figures';

times = [1:2:19];
shitTunFig = figure; 
pos(3)=3.3;  %17.4; %8.5;%pos(3)*1.5;
pos(4)=2.6;%pos(4)*1.5;
set(shitTunFig,'color','w','Units','inches','Position',pos)
hold on
plot(times',stabMat(:,1),'LineWidth',2,'Color',thisBlack)
plot(times',stabMat(:,2),'LineWidth',2,'Color',thisBlue)
plot(times',stabMat(:,3),'LineWidth',2,'Color',thisRed)
hold off
box on
% xlim([0,2500])
ylim([0,1])
lg = legend('Consistent','Switch','Loss');
set(lg,'FontSize',14)
xlabel('Days','FontSize',20)
ylabel('Fraction','FontSize',20)
set(gca,'FontSize',16,'LineWidth',1)

prefix = 'Tmaze_switch_tuning_Driscoll_fig3_0.15';
saveas(shitTunFig,[sFolder,filesep,prefix,'.fig'])
print('-depsc',[sFolder,filesep,prefix,'.eps'])
%  

%%
% heatmap of decoding weights
weightL = min(mat(:));
weightR = max(mat(:));
range = weightR - weightL;
ht = figure();
imagesc(mat)
colormap(rdmap);
ch = colorbar;
ch.Ticks = [weightL,weightL + round(range/2),weightR];
ch.TickLabels = {'-10','0','10'};

xlabel('Days')
ylabel('Neuron')
set(gca,'XTick',[1,3,8,11],'XTickLabel',{'1','5','15','21'})

prefix = 'Tmaze_switch_tuning_Driscoll_fig3_heatmap';
saveas(shitTunFig,[sFolder,filesep,prefix,'.fig'])
print('-depsc',[sFolder,filesep,prefix,'.eps'])
