% plot the figures related to T-maze model with both excitatory and
% inhibitory neruons

%% Define the parameters for graphics
% divergent colors, used in heatmap
nc = 256;           % number of colors
spectralMap = brewermap(nc,'Spectral');
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


set1 = brewermap(11,'Set1');
labelFontSize = 24;

%% 
saveFolder = '../figures';
dFile = '../data_in_paper/Tmaze_EI_06242022.mat';

load(dFile)

%%
%% Figures to show the drift, only excitatory neurons
% ========================================================
% population response for the Left and Right tasks, Fig 6B
% ========================================================
pvL = figure;
figureSize = [0 0 2.5 2.5];
set(pvL,'Units','inches','Position',figureSize,'PaperPositionMode','auto');

z = output.Yt(:,:,1);          % use the first time point data
L = size(z,2);
sel = max(z,[],2) > 0.05;
imagesc(z(sel,1:L/2),[0,0.3])

colorbar
xlabel('position','FontSize',16)
ylabel('neuron','FontSize',16)
ax = gca;
ax.XTick = [1 size(z,2)/2];
ax.XTickLabel = {'0', 'L'};
set(gca,'FontSize',16,'YTick','')

% prefix = 'Tmaze_EI_left_E_';
% saveas(pvL,[saveFolder,filesep,prefix,'.fig'])
% print('-depsc',[saveFolder,filesep,prefix,'.eps'])

pvR = figure;
figureSize = [0 0 2.5 2.5];
set(pvR,'Units','inches','Position',figureSize,'PaperPositionMode','auto');
imagesc(z(sel,(L/2+1):end),[0,0.3])

colorbar
xlabel('position','FontSize',20)
ylabel('neuron','FontSize',20)
ax = gca;
ax.XTick = [1 size(z,2)/2];
ax.XTickLabel = {'0', 'L'};
ax.YTickLabel = '';
set(gca,'FontSize',16,'YTick','')

% prefix = 'Tmaze_EI_right_E';
% saveas(pvR,[saveFolder,filesep,prefix,'.fig'])
% print('-depsc',[saveFolder,filesep,prefix,'.eps'])
% 
%%
% ============================================================
% Population vectors and similarity matrices, Fig 6B lower
% ============================================================
pvSM = figure; 
set(pvSM,'color','w','Units','inches')
pos(3)=4; pos(4)=2; 
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
xlabel('position','FontSize',20)
ylabel('neuron','FontSize',20)
% ax = gca;
aAxes.XTick = [1 size(z,2)/2];
aAxes.CLim = [0,0.3];
aAxes.XTickLabel = {'0', 'L'};
set(aAxes,'FontSize',16,'YDir','normal','YTick','')

imagesc(bAxes,selY(neurOrder,(1+L/2):L))

chPv = colorbar;
chPv.Position = [0.88,0.15,0.03,0.75];
xlabel('position','FontSize',20)
% ylabel('neuron','FontSize',20)
% ax = gca;
ylabel('')
bAxes.XTick = [1 size(z,2)/2];
bAxes.CLim = [0,0.3];
bAxes.XTickLabel = {'0', 'L'};
bAxes.YTickLabel = '';
set(bAxes,'FontSize',16,'YDir','normal','YTick','')

% prefix = 'tmaze_EI_PV_sorted';
% saveas(pvSM,[saveFolder,filesep,prefix,'.fig'])
% print('-depsc',[saveFolder,filesep,prefix,'.eps'])

%%
% ===========================================================
% A 2 by 2 heatmap array, similar to Fig 6C
% ===========================================================

pvGrid = figure; 
set(pvGrid,'color','w','Units','inches')
pos(3)=5; 
pos(4)=5;
set(pvGrid,'Position',pos)

% input and input similarity matrix
cwidth = 0.4; cheight = 0.37;
aAxes = axes('position',[.1  .5  cwidth  cheight]); hold on
bAxes = axes('position',[.55  .5  cwidth  cheight]);

cAxes = axes('position',[.1  .1  cwidth  cheight]); hold on
dAxes = axes('position',[.55  .1  cwidth  cheight]);
% fAxes = axes('position',[.65  .4  cwidth  cheight]);

% axes handles
axsh = cell(2,2);
axsh{1,1} = aAxes; axsh{1,2} = bAxes; 
axsh{2,1} = cAxes; axsh{2,2} = dAxes; 
% prepare the data
% allPVs = cell(3,3); % store the smilarity matrix ordered by different neuron index

inxSelPV = [1,100];
% inxSelPV = [100,250,500];

for i= 1:2
    % only selct those have significant tuning
    pkThd = 0.05;
    neuroInxL = max(output.Yt(:,1:size(z,2)/2,inxSelPV(i)),[],2) > pkThd;
    neuroInxR = max(output.Yt(:,size(z,2)/2+1:end,inxSelPV(i)),[],2) > pkThd;
    YtMerge = cat(1,output.Yt(neuroInxL,1:size(z,2)/2,:),output.Yt(neuroInxR,size(z,2)/2+1:end,:));
    [~,temp]= sort(YtMerge(:,:,inxSelPV(i)),2,'descend');
    [~, newInx] = sort(temp(:,1));
%     [~, newInx] = sort(mergePeakInx(:,inxSelPV(i)));
    for j = 1:length(inxSelPV)
        imagesc(axsh{i,j},YtMerge(newInx,:,inxSelPV(j)))
        set(axsh{i,j},'YDir','normal','FontSize',20)
%         colorbar
        axsh{i,j}.CLim = [0,0.3];
        axsh{i,j}.XLim = [0,size(z,2)/2];
        axsh{i,j}.XTick = [1 size(z,2)/2];
        axsh{i,j}.XTickLabel = {'0', 'L'};
        if j ~= 1
            axsh{i,j}.YTickLabel='';
        end
        if i ~=2
            axsh{i,j}.XTickLabel = '';
        end
    end
    
end

% add titile
title(axsh{1},'t = 1000', 'FontSize', 16)
title(axsh{3},'t = 5000', 'FontSize', 16)



% ======== representation similarity matrix =======
smFig = figure;
pos(3)=8; 
pos(4)=3;
set(smFig,'color','w','Units','inches','Position',pos)

for i = 1:length(inxSelPV)
%     [~,neuroInx] = sort(peakInx(:,inxSel(i)));
    SM = output.Yt(:,:,inxSelPV(i))'*output.Yt(:,:,inxSelPV(i));
    subplot(1,2,i)
    imagesc(SM,[0,0.45])
    colorbar
    ax = gca;
    ax.XTick = [1 500 1000];
    ax.YTick = [1 500 1000];
    ax.XTickLabel = {'0', 'L', ''};
    ax.YTickLabel = {'0', 'L', ''};
    title(['iteration', num2str(inxSelPV(i))],'FontSize',16)
    xlabel('position','FontSize',20)
    ylabel('position','FontSize',20)
end

% prefix = 'tmaze_EI_SM_compare';
% saveas(smFig,[saveFolder,filesep,prefix,'.fig'])
% print('-depsc',[saveFolder,filesep,prefix,'.eps'])

%% active fraction of neurons
activeRatio = nan(time_points,1);

for i=1:size(output.Yt,3)
    [pkCM, ~] = PlaceCellhelper.centerMassPks1D(output.Yt(:,:,i),0.05);
    actiInx = find(~isnan(pkCM));
    activeRatio(i) = length(actiInx)/param.NE;
end

% activeRatio = mean(~isnan(peakPosi),1);
h_af = figure;
set(h_af,'color','w','Units','inches','Position',[0,0,3.1,2.5])

plot((1:time_points)*param.record_step,activeRatio,'k','LineWidth',2)
xlabel('$\Delta t$','Interpreter','latex','FontSize',20)
ylabel('Active fraction','FontSize',20)
ylim([0,1])
xlim([0,5e3])
set(gca,'FontSize',16,'LineWidth',1,'XTick',[0,2500,5000])


%%
% ===========================================================
% A 2 by 2 heatmap array, Fig 6C
% ===========================================================

pvGrid = figure; 
set(pvGrid,'color','w','Units','inches')
pos(3)=5; 
pos(4)=5;
set(pvGrid,'Position',pos)

% input and input similarity matrix
cwidth = 0.4; cheight = 0.37;
aAxes = axes('position',[.1  .5  cwidth  cheight]); hold on
bAxes = axes('position',[.55  .5  cwidth  cheight]);

cAxes = axes('position',[.1  .1  cwidth  cheight]); hold on
dAxes = axes('position',[.55  .1  cwidth  cheight]);
% fAxes = axes('position',[.65  .4  cwidth  cheight]);

% axes handles
axsh = cell(2,2);
axsh{1,1} = aAxes; axsh{1,2} = bAxes; 
axsh{2,1} = cAxes; axsh{2,2} = dAxes; 
% prepare the data
allPVs = cell(3,3); % store the smilarity matrix ordered by different neuron index

inxSelPV = [1,200];
% inxSelPV = [100,250,500];
L = size(output.Yt,2)/2;  % total length of the track
for i= 1:2
    % only selct those have significant tuning
    pkThd = 0.05;
    neuroInxL = max(output.Yt(:,1:L,inxSelPV(i)),[],2) > pkThd;
    neuroInxR = max(output.Yt(:,L+1:end,inxSelPV(i)),[],2) > pkThd;
    YtMerge = cat(1,output.Yt(neuroInxL,1:L,:),output.Yt(neuroInxR,L+1:end,:));
    
    [pkCM, ~] = PlaceCellhelper.centerMassPks1D(YtMerge(:,:,inxSelPV(i)),0.05);
    actiInx = find(~isnan(pkCM));
    actiPkCM = pkCM(actiInx);
    [~,neurOrder] = sort(actiPkCM);
    newInx = actiInx(neurOrder);

    for j = 1:length(inxSelPV)
        imagesc(axsh{i,j},YtMerge(newInx,:,inxSelPV(j)),[0,0.3])
        set(axsh{i,j},'YDir','normal','FontSize',20)
%         colorbar
%         axsh{i,j}.CLim = [0,0.2];
        axsh{i,j}.XLim = [0,L];
        axsh{i,j}.XTick = [1 L];
        axsh{i,j}.XTickLabel = {'0', 'L'};
        if j ~= 1
            axsh{i,j}.YTickLabel='';
        end
        if i ~=2
            axsh{i,j}.XTickLabel = '';
        end
    end
    
end

% add titile
title(axsh{1},['t = ', num2str(inxSelPV(1))], 'FontSize', 16)
title(axsh{3},['t = ', num2str(inxSelPV(2))], 'FontSize', 16)

% ======== representation similarity matrix =======

smFig = figure;
pos(3)=8; 
pos(4)=3;
set(smFig,'color','w','Units','inches','Position',pos)

for i = 1:length(inxSelPV)
%     [~,neuroInx] = sort(peakInx(:,inxSel(i)));
    SM = output.Yt(:,:,inxSelPV(i))'*output.Yt(:,:,inxSelPV(i));
    subplot(1,2,i)
    imagesc(SM,[0,0.5])
    colorbar
    ax = gca;
    ax.XTick = [1 500 1000];
    ax.YTick = [1 500 1000];
    ax.XTickLabel = {'0', 'L', ''};
    ax.YTickLabel = {'0', 'L', ''};
    title(['iteration', num2str(inxSelPV(i))],'FontSize',16)
    xlabel('position','FontSize',20)
    ylabel('position','FontSize',20)
end

%% probability of shift

probShift = figure; 
set(probShift,'color','w','Units','inches')
pos(3)=3.4;  %17.4; %8.5;%pos(3)*1.5;
pos(4)=2.5;%pos(4)*1.5;
set(probShift,'Position',pos)
hold on
for i = 1:length(quantiles)
    plot(tps'*param.record_step,probs(:,i),'o-','MarkerSize',8,'Color',blues(1+3*i,:),...
        'MarkerFaceColor',blues(1+3*i,:),'MarkerEdgeColor',blues(1+3*i,:),'LineWidth',1.5)
end
hold off
box on

% lg = legend('\Delta s >1/4 L','\Delta s>1/2 L','\Delta s >3/4 L');
lg = legend('\Delta s > 0.07 L','\Delta s > 0.11 L','\Delta s > 0.22 L');
set(lg,'FontSize',10)
% ylim([0.0,0.6])
xlim([0,6e3])
xlabel('$\Delta t$','Interpreter','latex','FontSize',20)
ylabel({'Fraction of', 'peak moved'},'FontSize',20)
set(gca,'FontSize',16,'LineWidth',1,'XTick',[0,2500,5000])



%%
% ============================================================
% Analyze the fraction of neurons that change the tunning property: from
% left to right
% ============================================================

% summarize the figure
thisBlue = [52,153,204]/256;
thisRed = [218,28,92]/256;
thisBlack = [0,0,0];

times = (1:size(consist,1))*param.record_step;
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
xlim([0,5e3])
lg = legend('Lost','Gained');
set(lg,'FontSize',14)
xlabel('$\Delta t$','Interpreter','latex','FontSize',20)
ylabel('Fraction','FontSize',20)
set(gca,'FontSize',16,'LineWidth',1,'XTick',[0,2500,5000])

% prefix = 'Tmaze_switch_tuning';
% saveas(shitTunFig,[sFolder,filesep,prefix,'.fig'])
% print('-depsc',[saveFolder,filesep,prefix,'.eps'])