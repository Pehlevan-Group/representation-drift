% plot the figure for the drifting manifold tiling reuslts

%% Define the parameters for graphics
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
RdBuMap = flip(brewermap(nc,'RdBu'),1);
BlueMap = flip(brewermap(nc,'Blues'),1);


set1 = brewermap(11,'Set1');
labelFontSize = 24;


%% ============= Tmaze simulation ===========================

sFolder = './figures';
dFile = './data/Tmaze_1222.mat';

load(dFile)

% population response for the Left and Right tasks
pvL = figure;
figureSize = [0 0 2 2];
set(pvL,'Units','inches','Position',figureSize,'PaperPositionMode','auto');

z = Yt(:,:,1);  % use the first time point data
L = size(z,2);
sel = max(z,[],2) > 0.05;


% L = size(states_fixed_nn.Y,2);
% imagesc(states_fixed_nn.Y(:,1:L/2),[0,0.15])
imagesc(z(sel,1:L/2),[0,0.15])

colorbar
xlabel('position','FontSize',labelFontSize)
ylabel('neuron','FontSize',labelFontSize)
ax = gca;
ax.XTick = [1 500];
ax.XTickLabel = {'0', '\pi'};
set(gca,'FontSize',16,'YTick','')
prefix = 'tmazePV_nonoise_left';
% saveas(pvL,[sFolder,filesep,prefix,'.fig'])
% print('-depsc',[saveFolder,filesep,prefix,'.eps'])



pvR = figure;
figureSize = [0 0 2 2];
set(pvR,'Units','inches','Position',figureSize,'PaperPositionMode','auto');
% L = size(states_fixed_nn.Y,2);
% imagesc(states_fixed_nn.Y(:,(L/2+1):end),[0,0.15])
imagesc(z(sel,(L/2+1):end),[0,0.15])

colorbar
xlabel('position','FontSize',labelFontSize)
% ylabel('neuron','FontSize',20)
ax = gca;
ax.XTick = [1 500];
ax.XTickLabel = {'0', '\pi'};
ax.YTickLabel = '';
set(gca,'FontSize',16,'YTick','')
prefix = 'tmazePV_nonoise_right';
% saveas(pvR,[sFolder,filesep,prefix,'.fig'])
% print('-depsc',[saveFolder,filesep,prefix,'.eps'])


% =============== PV SM ====================
pvSM = figure; 
set(pvSM,'color','w','Units','inches')
pos(3)=4;  %17.4; %8.5;%pos(3)*1.5;
pos(4)=2;%pos(4)*1.5;
set(pvSM,'Position',pos)

% input and input similarity matrix
aAxes = axes('position',[.2  .15  0.3  0.8]); hold on
bAxes = axes('position',[.55  .15  0.3,0.8]); hold on

% [sortVal,sortedInx] = sort(states_fixed_nn.Y,2,'descend');
selY = z(sel,:);
[sortVal,sortedInx] = sort(selY,2,'descend');

[~,neurOrder] = sort(sortedInx(:,1));


% imagesc(aAxes,states_fixed_nn.Y(neurOrder,1:L/2))
imagesc(aAxes,selY(neurOrder,1:L/2))

% colorbar
xlabel('position','FontSize',labelFontSize)
ylabel('neuron','FontSize',labelFontSize)
% ax = gca;
aAxes.XTick = [1 500];
aAxes.CLim = [0,0.15];
aAxes.XTickLabel = {'0', '\pi'};
set(aAxes,'FontSize',16,'YDir','normal','YTick','')

% [sortVal,sortedInx] = sort(states_fixed_nn.Y,2,'descend');
% [~,neurOrder] = sort(sortedInx(:,1));
% imagesc(bAxes,states_fixed_nn.Y(neurOrder,(1+L/2):L))
imagesc(bAxes,selY(neurOrder,(1+L/2):L))

chPv = colorbar;
chPv.Position = [0.88,0.15,0.03,0.75];
xlabel('position','FontSize',labelFontSize)
% ylabel('neuron','FontSize',20)
% ax = gca;
ylabel('')
bAxes.XTick = [1 500];
bAxes.CLim = [0,0.15];
bAxes.XTickLabel = {'0', '\pi'};
bAxes.YTickLabel = '';
set(bAxes,'FontSize',16,'YDir','normal','YTick','')

prefix = 'tmazePV_sm';
saveas(pvSM,[sFolder,filesep,prefix,'.fig'])
print('-depsc',[saveFolder,filesep,prefix,'.eps'])


% ============= poulation vector before sorting =====

pvNoSort = figure; 
set(pvNoSort,'color','w','Units','inches')
pos(3)=4;  %17.4; %8.5;%pos(3)*1.5;
pos(4)=2;%pos(4)*1.5;
set(pvNoSort,'Position',pos)

% input and input similarity matrix
aAxes = axes('position',[.2  .15  0.3  0.8]);
bAxes = axes('position',[.55  .15  0.3,0.8]);

imagesc(aAxes,selY(:,1:L/2))
% colorbar
xlabel('position','FontSize',labelFontSize)
ylabel('neuron','FontSize',labelFontSize)
aAxes.XTick = [1 500];
aAxes.XTickLabel = {'0', '\pi'};
set(aAxes,'FontSize',16,'YDir','normal')

imagesc(bAxes,selY(:,(1+L/2):L))
chPv0 = colorbar;
chPv0.Position = [0.88,0.15,0.03,0.8];
aAxes.CLim = [0,0.15];
xlabel('position','FontSize',labelFontSize)
% ylabel('neuron','FontSize',20)
% ax = gca;
bAxes.CLim = [0,0.15];
bAxes.XTick = [1 500];
bAxes.XTickLabel = {'0', '\pi'};
bAxes.YTickLabel = '';
set(bAxes,'FontSize',16,'YDir','normal')

prefix = 'tmazePV_unsorted';
saveas(pvNoSort,[sFolder,filesep,prefix,'.fig'])
print('-depsc',[saveFolder,filesep,prefix,'.eps'])


% ================= A 3 by 3 heatmap array ================
pvGrid = figure; 
set(pvGrid,'color','w','Units','inches')
pos(3)=8.5;  %17.4; %8.5;%pos(3)*1.5;
pos(4)=7;%pos(4)*1.5;
set(pvGrid,'Position',pos)

% input and input similarity matrix
cwidth = 0.24; cheight = 0.27;
aAxes = axes('position',[.1  .7  cwidth  cheight]); hold on
bAxes = axes('position',[.37  .7  cwidth  cheight]);
cAxes = axes('position',[.65  .7  cwidth  cheight]);

dAxes = axes('position',[.1  .4  cwidth  cheight]); hold on
eAxes = axes('position',[.37  .4  cwidth  cheight]);
fAxes = axes('position',[.65  .4  cwidth  cheight]);


gAxes = axes('position',[.1  .08  cwidth  cheight]); hold on
hAxes = axes('position',[.37  .08  cwidth  cheight]);
iAxes = axes('position',[.65  .08  cwidth  cheight]);

% axes handles
axsh = cell(3,3);
axsh{1,1} = aAxes; axsh{1,2} = bAxes; axsh{1,3} = cAxes;
axsh{2,1} = dAxes; axsh{2,2} = eAxes; axsh{2,3} = fAxes;
axsh{3,1} = gAxes; axsh{3,2} = hAxes; axsh{3,3} = iAxes;

% prepare the data
allPVs = cell(3,3); % store the smilarity matrix ordered by different neuron index

inxSelPV = [20,100,400];
% inxSelPV = [100,250,500];

for i= 1:3
    % only selct those have significant tuning
    pkThd = 0.05;
    neuroInxL = max(Yt(:,1:500,inxSelPV(i)),[],2) > pkThd;
    neuroInxR = max(Yt(:,501:1000,inxSelPV(i)),[],2) > pkThd;
    YtMerge = cat(1,Yt(neuroInxL,1:500,:),Yt(neuroInxR,501:1000,:));
%     [~,neuroInx] = sort(peakInx(:,inxSelPV(i)));

    % merged left and right
%     YtMerge = cat(1,Yt(neuroInx(1:round(k/2)),1:500,:),Yt(neuroInx((k/2+1):end),501:1000,:));
%     YtMerge = cat(1,Yt(neuroInx(1:round(k/2)),1:500,:),Yt(neuroInx((k/2+1):end),501:1000,:));
%     mergeInx = nan(k,time_points);
%     
%     for j = 1:time_points
%         [~,mergePeakPosi] = sort(YtMerge(:,:,j),2,'descend');
%         mergePeakInx(:,j) = mergePeakPosi(:,1);
%     end
    [~,temp]= sort(YtMerge(:,:,inxSelPV(i)),2,'descend');
    [~, newInx] = sort(temp(:,1));
%     [~, newInx] = sort(mergePeakInx(:,inxSelPV(i)));
    for j = 1:length(inxSelPV)
%         axes(axsh{i,j})
%         imagesc(axsh{i,j},YtMerge(newInx(100:400),:,inxSelPV(j)))
        imagesc(axsh{i,j},YtMerge(newInx,:,inxSelPV(j)))
%         imagesc(axsh{i,j},YtMerge(newInx(301:400),:,inxSelPV(j)))
        set(axsh{i,j},'YDir','normal','FontSize',20)
%         colorbar
        axsh{i,j}.CLim = [0,0.2];
        axsh{i,j}.XLim = [0,500];
        axsh{i,j}.XTick = [1 500];
        axsh{i,j}.XTickLabel = {'0', '\pi'};
        if j ~= 1
            axsh{i,j}.YTickLabel='';
        end
        if i ~=3
            axsh{i,j}.XTickLabel = '';
        end
%         title(['iteration ', num2str(inxSelPV(j))])
%         xlabel('position')
%         ylabel('sorted index')
    end
    
end
% colormap(BlueMap)
ch3 = colorbar;
ch3.Position = [0.9,0.1,0.02,cheight];


prefix = 'tmazePV_ordered';
saveas(pvGrid,[saveFolder,filesep,prefix,'.fig'])
print('-depsc',[saveFolder,filesep,prefix,'.eps'])

% figure
% imagesc(YtMerge(newInx(100:400),:,inxSelPV(j)),[0,0.4])
% colorbar

% ================ RF shift of individual cells =============
% we select three neurons to see how their RFs chagne 
% epLoc = randperm(400,3);

rfShift = figure; 
set(rfShift,'color','w','Units','inches')
pos(3)=4;  %17.4; %8.5;%pos(3)*1.5;
pos(4)=2.5;%pos(4)*1.5;
set(rfShift,'Position',pos)
labelFontSize = 24;
gcaFontSize = 20;

% input and input similarity matrix
cwidth = 0.25; cheight = 0.7;
aAxes = axes('position',[.15  .25 cwidth  cheight]);
bAxes = axes('position',[.42  .25  cwidth  cheight]);
cAxes = axes('position',[.72  .25  cwidth  cheight]);


axhdls = cell(3,1);
axhdls{1} = aAxes; axhdls{2} = bAxes; axhdls{3} = cAxes;

epLoc = [82, 302, 129];  % this should b carefully chosen
for i = 1:3
    ypart = squeeze(Yt(epLoc(i),1:500,:));
    axes(axhdls{i})
    imagesc(ypart')
%     colorbar
    xlabel('location')
    if i==1
        ylabel('time')
    end
    set(gca,'XTick',[0, 250, 500],'XTickLabel',{'0','0.5','1'})
    if y ~=1
        set(gca,'YTickLabel','')
    end
%     title(['neuron ',num2str(epLoc(i))])
end

% save the figure
prefix = 'tmaze_RF_example';
saveas(rfShift,[sFolder,filesep,prefix,'.fig'])
print('-depsc',[saveFolder,filesep,prefix,'.eps'])



% ======== representation similarity matrix =======
smFig = figure;
pos(3)=11;  %17.4; %8.5;%pos(3)*1.5;
pos(4)=3;%pos(4)*1.5
set(smFig,'color','w','Units','inches','Position',pos)

for i = 1:length(inxSel)
%     [~,neuroInx] = sort(peakInx(:,inxSel(i)));
    SM = Yt(:,:,inxSel(i))'*Yt(:,:,inxSel(i));
    subplot(1,3,i)
    imagesc(SM,[0,0.5])
    colorbar
    ax = gca;
    ax.XTick = [1 500 1000];
    ax.YTick = [1 500 1000];
    ax.XTickLabel = {'0', '\pi', '2\pi'};
    ax.YTickLabel = {'0', '\pi', '2\pi'};
    title(['iteration', num2str(inxSel(i))],'FontSize',20)
    xlabel('position','FontSize',20)
    ylabel('position','FontSize',20)
end
prefix = 'tmaze_SM_compare';
saveas(smFig,[sFolder,filesep,prefix,'.fig'])
print('-depsc',[saveFolder,filesep,prefix,'.eps'])


%% ============ Gain and loss of RF ====================
pkThd = 0.05;     % peak threshold to determine if a neuron has RF
[k, N, time_points] = size(Yt);
            % peak of receptive field
peakInxLeft = nan(k,time_points);
peakValLeft = nan(k,time_points);
peakInxRight = nan(k,time_points);
peakValRight = nan(k,time_points);
for i = 1:time_points
    [pkValL, peakPosiL] = sort(Yt(:,1:round(N/2),i),2,'descend');
    [pkValR, peakPosiR] = sort(Yt(:,(1+round(N/2)):N,i),2,'descend');
    peakInxLeft(:,i) = peakPosiL(:,1);
    peakValLeft(:,i) = pkValL(:,1);
    peakInxRight(:,i) = peakPosiR(:,1);
    peakValRight(:,i) = pkValR(:,1);
end

% fraction of neurons that are active at any given time
% quantified by the peak value larger than a threshold 0.01
rfIndexLeft = peakValLeft > pkThd;
rfIndexRight = peakValRight > pkThd;

rfIndex = rfIndexLeft | rfIndexRight;

% fraction of neurons
activeRatioLeft = sum(rfIndexLeft,1)/k;
activeRatioRight = sum(rfIndexRight,1)/k;
activeRatio = sum(rfIndex,1)/k;
activeRatioBoth = sum(rfIndexLeft & rfIndexRight,1)/k;

peakStats.acRatioL = activeRatioLeft;
peakStats.acRatioR = activeRatioRight;
peakStats.actRatio = activeRatio;
peakStats.actRatioB = peakStats.actRatio;

% select a start index from which the dynamics can be regarded
% as stationary
stInx = 101;
% plot the figure
dropInOut = figure;
subplot(2,2,1)
plot(stInx:time_points,activeRatio(stInx:end))
ylim([0,1])
xlabel('Iteration')
ylabel('Active fraction')
title('Left or Right')

subplot(2,2,2)
plot(stInx:time_points,activeRatioBoth(stInx:end))
ylim([0,1])
xlabel('Iteration')
ylabel('Active fraction')
title('Left and Right')

subplot(2,2,3)
plot(stInx:time_points,activeRatioLeft(stInx:end))
ylim([0,1])
xlabel('Iteration')
ylabel('Active fraction')
title('Left only')

subplot(2,2,4)
plot(stInx:time_points,activeRatioRight(stInx:end))
ylim([0,1])
xlabel('Iteration')
ylabel('Active fraction')
title('Right only')

prefix = 'dropin_out_statistics';
saveas(dropInOut,[sFolder,filesep,prefix,'.fig'])
print('-depsc',[saveFolder,filesep,prefix,'.eps'])


% ============================================================
% Analyze the fraction of neurons that change the tunning property: from
% left to right
% ============================================================
tuningLR = nan(k,time_points);
peakInx = nan(k,time_points);
% peakInxLeft = nan(k,time_points);
% peakValLeft = nan(k,time_points);
% peakInxRight = nan(k,time_points);
% peakValRight = nan(k,time_points);
for i = 1:time_points
    [pkVal, peakPosi] = sort(Yt(:,:,i),2,'descend');
    actInx = pkVal(:,1)>pkThd;
    peakInx(actInx,i) = peakPosi(actInx,1);
end

% statistics of the tuning change
timedT  = round(time_points/2); % maximum seperation of time
consist = nan(timedT,2);  % mean and standard deviation
shiftTun = nan(timedT,2);
lossTun = nan(timedT,2);
gainTun = nan(timedT,2);
tuningInfo = nan(k,time_points);
tuningInfo(peakInx <= round(N/2)) = -1;  % left tuning
tuningInfo(peakInx > round(N/2)) = 1;    % right tuning

for dt = 1:round(time_points/2)
    % consistent
    tempC = tuningInfo(:,1+dt:time_points) - tuningInfo(:,1:time_points-dt);
    refTuning = mean(~isnan(tuningInfo(:,1:time_points-dt)),1);
    consist(dt,:) = [mean(mean(tempC==0,1)./refTuning),std(mean(tempC==0,1)./refTuning)];
    
    % shifted
    shiftTun(dt,:) = [mean(mean(abs(tempC) > 0,1)./refTuning),std(mean(abs(tempC) > 0,1)./refTuning)];
    
    % loss of tuning
    tmp1 = ~isnan(tuningInfo(:,1:time_points-dt));
    tmp2 = isnan(tuningInfo(:,1+dt:time_points));
    lossTun(dt,:) = [mean(mean(tmp1.*tmp2,1)./refTuning),std(mean(tmp1.*tmp2,1)./refTuning)];
    
    % gain of tuning
    refTuning = mean(isnan(tuningInfo(:,1:time_points-dt)),1);
    tmp1 = isnan(tuningInfo(:,1:time_points-dt));
    tmp2 = ~isnan(tuningInfo(:,1+dt:time_points));
    gainTun(dt,:) = [mean(mean(tmp1.*tmp2,1)./refTuning),std(mean(tmp1.*tmp2,1)./refTuning)];
end


% summarize the figure
thisBlue = [52,153,204]/256;
thisRed = [218,28,92]/256;
thisBlack = [0,0,0];

timeSep = 10;
times = (1:size(consist,1))*timeSep;
shitTunFig = figure; 
pos(3)=3.1;  %17.4; %8.5;%pos(3)*1.5;
pos(4)=2.5;%pos(4)*1.5;
set(shitTunFig,'color','w','Units','inches','Position',pos)
hold on
% plot(times',consist(:,1),'LineWidth',2,'Color',thisBlack)
% plot(times',shiftTun(:,1),'LineWidth',2,'Color',thisRed)
plot(times',lossTun(:,1),'LineWidth',2,'Color',thisRed)
plot(times',gainTun(:,1),'LineWidth',2,'Color',thisBlue)
hold off
box on
xlim([0,400])
lg = legend('Lost','Gained');
set(lg,'FontSize',14)
xlabel('$\Delta t$','Interpreter','latex','FontSize',20)
ylabel('Fraction','FontSize',20)
set(gca,'FontSize',16,'LineWidth',1)

prefix = 'Tmaze_switch_tuning';
saveas(shitTunFig,[sFolder,filesep,prefix,'.fig'])
print('-depsc',[saveFolder,filesep,prefix,'.eps'])


% ******************************************************
% plot figure showing the consistent and shifted
% ******************************************************
shitTunFig = figure; 
pos(3)=3.8;  %17.4; %8.5;%pos(3)*1.5;
pos(4)=3;%pos(4)*1.5;
set(shitTunFig,'color','w','Units','inches','Position',pos)
hold on
plot(times',consist(:,1),'LineWidth',2,'Color',thisBlack)
plot(times',lossTun(:,1),'LineWidth',2,'Color',thisRed)
plot(times',gainTun(:,1),'LineWidth',2,'Color',thisBlue)
plot(times',shiftTun(:,1),'LineWidth',2,'Color',greys(6,:))

hold off
box on
xlim([0,1000])
lg = legend('Consistent','Lost','Gained','Shift');
set(lg,'FontSize',14)
xlabel('$\Delta t$','Interpreter','latex','FontSize',20)
ylabel('Fraction','FontSize',20)
set(gca,'FontSize',16,'LineWidth',1)

prefix = 'Tmaze_tuning_stability_statistics';
saveas(shitTunFig,[saveFolder,filesep,prefix,'.fig'])
print('-depsc',[saveFolder,filesep,prefix,'.eps'])

 
% LeftTunNeurons = peakInx < round(N/2);
% figure
% imagesc(LeftTunNeurons)
% % fraction from left to right
% day1Lelft = peakInx(:,1) <= round(N/2);
% day1Right = peakInx(:,1) > round(N/2);
% 
% shiftL2R = peakInx(day1Lelft,:) > round(N/2);
% figure
% imagesc(shiftL2R)
% figure
% plot(mean(shiftL2R,1))
% xlabel('Time')
% ylabel('Fraction')

% fraction from right to left
% RightTunNeurons = peakInx > round(N/2);
% figure
% imagesc(RightTunNeurons)
% 
% shiftR2L = peakInx(day1Right,:) < round(N/2);
% figure
% plot(mean(shiftR2L,1))
% xlabel('Time')
% ylabel('Fraction')

% RightTunNeurons = peakInx > round(N/2);
% figure
% imagesc(RightTunNeurons)
% 
% % all shifted fraction over time
% figure
% plot(mean([mean(shiftL2R,1);mean(shiftR2L,1)],1))

% ============== total active ratio ==================
% plot the total active ratio, for manuscript
totActiveRatio = figure; 
set(totActiveRatio,'color','w','Units','inches')
pos(3)=3.4;  %17.4; %8.5;%pos(3)*1.5;
pos(4)=2.5;%pos(4)*1.5;
set(totActiveRatio,'Position',pos)
plot(((stInx:time_points)-stInx)*step,activeRatio(stInx:end),'LineWidth',2)
ylim([0,1])
xlim([0,400])
xlabel('$\Delta t$','Interpreter','latex','FontSize',20)
ylabel({'Fraction of cells','with active RF'},'FontSize',20)
set(gca,'FontSize',16,'XTick',(0:2:4)*100,'LineWidth',1)
% set(gca,'FontSize',16,'XTick',[0,500,1000,1500])
% title('Left or Right')
prefix = 'Tmaze_activeRatio';
saveas(totActiveRatio,[saveFolder,filesep,prefix,'.fig'])
print('-depsc',[saveFolder,filesep,prefix,'.eps'])


% ================ drop in and out heat maps =========

% heatmap showing whether a neuron is active or silent ordered
% by the time point 0
[~,neuroInxLeft] = sort(rfIndexLeft(:,stInx),'descend');
[~,neuroInxRight] = sort(rfIndexRight(:,stInx),'ascend');

gys = brewermap(256,'Greys');
% greyMap = flip(gys(1:200,:),1);
greyMap = gys(1:200,:);


LtoR = figure;
set(LtoR,'color','w','Units','inches')
pos(3)=4;  %17.4; %8.5;%pos(3)*1.5;
pos(4)=2.8;%pos(4)*1.5;
set(LtoR,'Position',pos)
% subplot(1,2,1)
imagesc(rfIndexLeft(neuroInxLeft,stInx:end))
colormap(greyMap)
xlabel('$\Delta t$','Interpreter','latex')
ylabel('Sorted index')
prefix = 'Tmaze_LtoR';
saveas(LtoR,[sFolder,filesep,prefix,'.fig'])
print('-depsc',[saveFolder,filesep,prefix,'.eps'])

% title('Active for Left Trials')

% subplot(1,2,2)
figure
imagesc(rfIndexRight(neuroInxRight,stInx:end))
xlabel('Iteration')
ylabel('Ordered neuron index')
title('Active for Right Trials')



% ================  Gain and loss of RF =============
%
sepInx = find(rfIndexLeft(neuroInxLeft,stInx)==0,1);
leftLoss = 1 - sum(rfIndexLeft(neuroInxLeft(1:(sepInx-1)),stInx:end),1)/(sepInx-1);
leftGain = sum(rfIndexLeft(neuroInxLeft(sepInx:end),stInx:end),1)/(k-sepInx+1);

sepInxR = find(rfIndexRight(neuroInxRight,stInx)==1,1);
RightLoss = 1 - sum(rfIndexRight(neuroInxRight(sepInxR:end),stInx:end),1)/(k-sepInxR+1);
RightGain = sum(rfIndexRight(neuroInxRight(1:(sepInxR-1)),stInx:end),1)/(sepInxR-1);

gainLoss = figure;
set(gainLoss,'color','w','Units','inches')
pos(3)=4;  %17.4; %8.5;%pos(3)*1.5;
pos(4)=2.8;%pos(4)*1.5;
set(gainLoss,'Position',pos)
% subplot(1,2,1)
plot((1:length(leftLoss))*step,leftLoss)
hold on
plot((1:length(leftLoss))*step,leftGain)
% ylim([0,1])
legend('loss','gain','Location','northwest')
xlabel('$\Delta t$','Interpreter','latex','FontSize',20)
ylabel('Fraction','FontSize',20)
set(gca,'LineWidth',1,'FontSize',16)

prefix = 'Tmaze_gainLoss';
saveas(gainLoss,[sFolder,filesep,prefix,'.fig'])
print('-depsc',[saveFolder,filesep,prefix,'.eps'])



% subplot(1,2,2)
figure
plot(RightLoss)
hold on
plot(RightGain)
legend('loss: right','gain:right')
xlabel('Iteration')
ylabel('Fraction')
hold off  
prefix = 'Tmaze_gainLoss_Right';
saveas(gcf,[sFolder,filesep,prefix,'.fig'])
print('-depsc',[saveFolder,filesep,prefix,'.eps'])


%% Probability of shift of peak positions

psRange = pi;   % the range of ring positions
pks0 = orderedPeaks/1001*psRange;
tps = [5,10,15,20];
% tps = [5,10,20,50,100];
% quantiles = (1:3)/10*psRange;
quantiles = [0.35,0.5,1]/4.5*psRange;
probs = nan(length(tps),length(quantiles));
for i = 1:length(tps)
    diffLag = min(pi, abs(pks0(:,tps(i)+1:end,:) - pks0(:,1:end-tps(i),:)));
    for j = 1:length(quantiles)
        probs(i,j) = sum(diffLag(:)>quantiles(j))/length(diffLag(:));
    end
end


probShift = figure; 
set(probShift,'color','w','Units','inches')
pos(3)=3.4;  %17.4; %8.5;%pos(3)*1.5;
pos(4)=2.5;%pos(4)*1.5;
set(probShift,'Position',pos)
hold on
for i = 1:length(quantiles)
    plot(tps'*step,probs(:,i),'o-','MarkerSize',8,'Color',blues(1+3*i,:),...
        'MarkerFaceColor',blues(1+3*i,:),'MarkerEdgeColor',blues(1+3*i,:),'LineWidth',1.5)
end
hold off
box on

% lg = legend('\Delta s >1/4 L','\Delta s>1/2 L','\Delta s >3/4 L');
lg = legend('\Delta s > 0.07 L','\Delta s > 0.11 L','\Delta s > 0.22 L');
% lg = legend('\Delta s > 0.1 L','\Delta s > 0.2 L','\Delta s > 0.3 L');
set(lg,'FontSize',12)
xlabel('$\Delta t$','Interpreter','latex','FontSize',20)
ylabel({'Fraction of', 'peak moved'},'FontSize',20)
set(gca,'LineWidth',1,'FontSize',16)
ylim([0.2,0.6])
xlim([0,450])
prefix = 'Tmaze_prob_shift_dist';
saveas(gcf,[saveFolder,filesep,prefix,'.fig'])
print('-depsc',[saveFolder,filesep,prefix,'.eps'])

%% Shift a RF, more strigent critieor for active

% based on unsorted data
pkThreshold = 0.05;
psRange = 2*pi;
pks0 = peakInx/1001*psRange;   % for ring input, 2pi, for Tmaze pi
activeFlag = peakVal > pkThreshold;  % active or silent
pks_masked = pks0.*double(activeFlag);
pks_masked(pks_masked==0) = nan;
% tps = [10,100,200];
% tps = [5,10,20,35,50];
tps = [5,10,20,50,100,200];

% quantiles = (1:3)/4*pi;
quantiles = (1:3)/10*psRange;
% quantiles = [0.05,0.15,0.25]*psRange;

probs = nan(length(tps),length(quantiles));


shifDist = figure; 
set(shifDist,'color','w','Units','inches')
pos(3)=3.2;  %17.4; %8.5;%pos(3)*1.5;
pos(4)=3.5;%pos(4)*1.5;
set(shifDist,'Position',pos)
hold on
for i = 1:length(tps)
    temp = pks_masked(:,tps(i)+1:end,:) - pks_masked(:,1:end-tps(i),:);
    [f,xi] = ksdensity(temp(~isnan(temp))); 
%     histogram(temp(~isnan(temp)),'Normalization','pdf')
%     plot(xi,f,'Color',blues(1+3*i,:),'LineWidth',2)
    plot(xi,f,'LineWidth',2)
    diffLag = min(pi, abs(temp(~isnan(temp))));
    
    for j = 1:length(quantiles)
        probs(i,j) = sum(diffLag(:)>quantiles(j))/length(diffLag(:));
    end
end

% compared with random position
randPks = nan(size(pks_masked));
for i = 1:size(pks_masked,2)
    inx = find(~isnan(pks_masked(:,i)));
    randPks(inx,i) = psRange*rand(length(inx),1);
end
temp = pks_masked - randPks;
diffLagRnd = min(pi, abs(temp(~isnan(temp))));
[f,xi] = ksdensity(temp(~isnan(temp))); 
plot(xi,f,'Color',greys(8,:),'LineWidth',2)
hold off
box on
xlim([-6.5,6.5])
lg = legend('\Delta t=5','\Delta t=20','\Delta t=50','Random','Location','northeast');
set(lg,'FontSize',14)
xlabel('$\Delta s$','Interpreter','latex','FontSize',24)
ylabel('pdf','FontSize',24)
set(gca,'LineWidth',1,'FontSize',20)



probShift = figure; 
set(probShift,'color','w','Units','inches')
pos(3)=3.2;  %17.4; %8.5;%pos(3)*1.5;
pos(4)=3.5;%pos(4)*1.5;
set(probShift,'Position',pos)
hold on
for i = 1:length(quantiles)
    plot(tps'*step,probs(:,i),'o-','MarkerSize',8,'Color',blues(1+3*i,:),...
        'MarkerFaceColor',blues(1+3*i,:),'MarkerEdgeColor',blues(1+3*i,:),'LineWidth',1.5)
end
hold off
box on

lg = legend('\Delta s > 0.05 L','\Delta s > 0.15 L','\Delta s > 0.25 L');
set(lg,'FontSize',14)
xlabel('$\Delta t$','Interpreter','latex','FontSize',20)
ylabel({'Fraction of','peak moved'},'FontSize',20)
set(gca,'LineWidth',1,'FontSize',16)


% probabilty of shift interval that are less than 10%
tps = 1:2:200;
probs10 = nan(length(tps),1);
for i = 1:length(tps)
    temp = pks_masked(:,tps(i)+1:end,:) - pks_masked(:,1:end-tps(i),:);
%     [f,xi] = ksdensity(temp(~isnan(temp))); 
%     histogram(temp(~isnan(temp)),'Normalization','pdf')
%     plot(xi,f,'Color',set1(i,:),'LineWidth',3)
    diffLag = min(pi, abs(temp(~isnan(temp))));
    probs10(i) = sum(diffLag(:)<0.1*pi)/length(diffLag(:));
end

% random
probRnd = sum(diffLagRnd(:)<0.1*pi)/length(diffLagRnd(:));

figure
hold on
plot(tps',probs10,'LineWidth',4)
plot([0;200],[probRnd;probRnd],'k--','LineWidth',4)
hold off
box on
legend('Model','Random')
xlabel('Time','FontSize',28)
ylabel('Fraction < 10% shift','FontSize',28)
set(gca,'LineWidth',1,'FontSize',24)

%% Animation of the peak position along the semi-ring
% randomly select 10 neurons
selNeur = randperm(size(pks_masked,1),10);
set1 = brewermap(11,'Set1');
frameSep = 3; % only capture every 5 steps, each step  = 2 iterations

epPosiXs = cos(pks_masked(selNeur,:));
epPosiYs = sin(pks_masked(selNeur,:)); 

thetas = 0:0.05:2*pi;
figure
plot(cos(thetas),sin(thetas),'LineWidth',1.5,'Color',greys(5,:));
hold on
for i = 1:10
    plot(epPosiXs(i,1),epPosiYs(i,1),'o','MarkerSize',20,'MarkerEdgeColor',...
        set1(i,:),'LineWidth',3);
    xlim([-1,1])
    ylim([0,1])
    text(epPosiXs(i,1)-0.02,epPosiYs(i,1),num2str(i),'FontSize',20)
end
hold off
box on
ax = gca;
ax.XAxis.Visible = 'off';
ax.YAxis.Visible = 'off';
xlabel('x','FontSize',20)
ylabel('y','FontSize',20)
% set(gca,'LineWidth',1.5,'FontSize',20)

detlaT = 0.01;
f = getframe(gcf);
[im,map] = rgb2ind(f.cdata,256,'nodither');
k = 1;
for i = 1:round(size(epPosiXs,2)/frameSep)
    plot(cos(thetas),sin(thetas),'LineWidth',1.5,'Color',greys(5,:));
    hold on
    for j=1:10
        plot(epPosiXs(j,i*frameSep),epPosiYs(j,i*frameSep),'o','MarkerSize',20,'MarkerEdgeColor',...
        set1(j,:),'LineWidth',3)
        text(epPosiXs(j,i*frameSep)-0.02,epPosiYs(j,i*frameSep),num2str(j),'FontSize',16)
    end
    hold off
    box on
    xlim([-1,1])
    ylim([-1,1])
    ax = gca;
    ax.XAxis.Visible = 'off';
    ax.YAxis.Visible = 'off';
    xlabel('x','FontSize',20)
    ylabel('y','FontSize',20)
    set(gca,'LineWidth',1.5,'FontSize',20)
    title(['time = ', num2str(i*frameSep)])
    f = getframe(gcf);
    im(:,:,1,k) = rgb2ind(f.cdata,map,'nodither');
    k = k + 1;
    
end
imwrite(im,map,'scatterPkPosiRing.gif','DelayTime',detlaT,'LoopCount',inf)

%% 
% time period that a neuron has active place field
temp = ~isnan(pks_masked);
tolActiTime = sum(temp,2)/size(pks_masked,2);
avePks = nan(size(pks_masked));
avePks(temp) = pks_masked(temp);
meanPks = nanmean(avePks,2);

figure
plot(meanPks,tolActiTime,'o','MarkerSize',8,'LineWidth',2)
xlabel('average peak amplitude','FontSize',24)
ylabel('faction of active time','FontSize',24)
set(gca,'FontSize',20,'LineWidth',1.5)