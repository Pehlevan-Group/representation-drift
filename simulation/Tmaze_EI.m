% this program run the simulation of Tmaze with both excitatory and
% inhibitory neurons

close all
clear

%% setting for the graphics
% plot setting
defaultGraphicsSetttings
blues = brewermap(11,'Blues');
rb = brewermap(11,'RdBu');
set1 = brewermap(9,'Set1');

saveFolder = './figures';
%% Generate sample data

param.NE = 200;         % number of excitatory neurons
param.NI = 20;          % number of inhibitory
param.Nx = 3;           % input dimension
param.noise = 5e-4; 
param.learnRate = 1e-2;
param.alpha = 1.6;
param.lbd1 = 5e-6;      % regularization for the simple nsm, 1e-3
param.lbd2 = 1e-3;      % default 1e-3
param.BatchSize = 1;
param.gy = 0.05;        % update step for y
param.gz = 0.05;        % update step for z


param.record_step = 1000;  % record neural activity every 100 steps

% initialize weight matrices and bias
param.W = 0.5*randn(param.NE,param.Nx);    % initialize the forward matrix
param.Wei = 0.05*rand(param.NE,param.NI);  % i to e connection
param.Wie = param.Wei';                    % e to i connections
param.M = eye(param.NI);                   % recurrent interaction between inhibiotry neurons
param.b = zeros(param.NE,1);  % biase


% generate input data
t = 1e3;    % total number of samples previously 1e4

radius = 1;
sep = 2*pi/t;
X1 = [radius*cos(0:sep:pi-sep);radius*sin(0:sep:pi-sep)]; 
X = [X1,X1;ones(1,t/2),-ones(1,t/2)];
Xsel = X(:,4:4:end);  % subsampling to reduce simulation time


% plot input similarity matrix
SMX = X'*X;


%% Initial transient stage
total_iter = 1e4;
record_flag = true;  % record or not
[output, param] = Tmaze_EI_update(X,Xsel, total_iter, param, record_flag);

%% Analysis of the learned representations and weights

% first, figure out the tuning properties of E and I neurons
Ys = output.Yt(:,:,end);
Zs = output.Zt(:,:,end);
[pkCM_E, ~] = PlaceCellhelper.centerMassPks1D(Ys,0.05);
[pkCM_I, ~] = PlaceCellhelper.centerMassPks1D(Zs,0.05);


% left turn and right turn
L = size(output.Yt,2)/2;
Rinx_E = pkCM_E < L;
Linx_E = pkCM_E > L;
Rinx_I = pkCM_I < L;
Linx_I = pkCM_I > L;

% Sort the receptive fields for E and I neurons
actiInx_E = find(~isnan(pkCM_E));
actiPkCM_E = pkCM_E(actiInx_E);
[~,neurOrder_E] = sort(actiPkCM_E);

actiInx_I = find(~isnan(pkCM_I));
actiPkCM_I = pkCM_I(actiInx_I);
[~,neurOrder_I] = sort(actiPkCM_I);

% ************************************************************
% visualize learned representations
% ************************************************************
% excitatory neurons
figE = figure;
figureSize = [0 0 3.6 2.7];
set(figE,'Units','inches','Position',figureSize,'PaperPositionMode','auto');

imagesc(Ys(actiInx_E(neurOrder_E),:),[0,0.3])
% colormap(viridis)
colorbar
xlabel('Position')
ylabel('Neuron sorted')
title('Excitatory neurons')

% prefix = ['Tmaze_', 'EI_Ye_06222022'];
% saveas(gcf,[saveFolder,filesep,prefix,'.fig'])
% print('-depsc',[saveFolder,filesep,prefix,'.eps'])

% inhibitory neurons
figI = figure;
figureSize = [0 0 3.6 2.7];
set(figI,'Units','inches','Position',figureSize,'PaperPositionMode','auto');

imagesc(Zs(actiInx_I(neurOrder_I),:),[0,0.5])
% colormap(viridis)
colorbar
xlabel('Position')
ylabel('Neuron sorted')
title('Inhibitory neurons')

% prefix = ['Tmaze_', 'EI_Yi_06222022'];
% saveas(gcf,[saveFolder,filesep,prefix,'.fig'])
% print('-depsc',[saveFolder,filesep,prefix,'.eps'])


% ************************************************************
% visualize the weight matrices based on the ordering
% ************************************************************
temp = param.Wei(actiInx_E(neurOrder_E),:);
Wei_order = temp(:,actiInx_I(neurOrder_I));

figure
imagesc(Wei_order,[0,0.015])
% colormap(viridis)
colorbar
xlabel('I neuron index')
ylabel('E neuron index')

% prefix = ['Tmaze_', 'EI_wie_06212022'];
% saveas(gcf,[saveFolder,filesep,prefix,'.fig'])
% print('-depsc',[saveFolder,filesep,prefix,'.eps'])

% ************************************************************
% visualize the weight matrices based on the ordering
% ************************************************************
temp = param.M(actiInx_I(neurOrder_I),:);
M_order = temp(:,actiInx_I(neurOrder_I));
figure
imagesc(M_order,[0,0.03])
% colormap(viridis)
colorbar
xlabel('I neuron index')
ylabel('I neuron index')

% save the figure
% prefix = ['Tmaze_', 'EI_wii_06212022'];
% saveas(gcf,[saveFolder,filesep,prefix,'.fig'])
% print('-depsc',[saveFolder,filesep,prefix,'.eps'])


%% Continuous update to show the drift
total_iter = 1e4;
param.record_step = 50;
[output, param] = Tmaze_EI_update(X,Xsel, total_iter, param, record_flag);

time_points = round(total_iter/param.record_step);


%% Figures to show the drift, only excitatory neurons
% ========================================================
% population response for the Left and Right tasks, Fig 6B
% ========================================================
pvL = figure;
figureSize = [0 0 3 3];
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
figureSize = [0 0 3 3];
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
allPVs = cell(3,3); % store the smilarity matrix ordered by different neuron index

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
    imagesc(SM)
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
figure
plot(1:time_points,activeRatio)
xlabel('iteration')
ylabel('active fraction')
ylim([0,1])


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

inxSelPV = [1,100];
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
        imagesc(axsh{i,j},YtMerge(newInx,:,inxSelPV(j)),[0,0.5])
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

% first, statistics of the centroids
centroidPosi = nan(param.NE,time_points);
for i = 1:time_points
    [pkCM, ~] = PlaceCellhelper.centerMassPks1D(output.Yt(:,:,i),0.05);
    centroidPosi(:,i) = pkCM;
end

adjPeaks = mod(centroidPosi, size(output.Yt,2)/2);
psRange = pi;   % the range of ring positions
pks0 = adjPeaks/size(output.Yt,2)*2*psRange;
% tps = [1,2,3,8];
tps = [20,40,60,80,100];
% quantiles = (1:3)/10*psRange;
quantiles = [0.35,0.5,1]/4.5*psRange;
probs = nan(length(tps),length(quantiles));
for i = 1:length(tps)
%     diffLag = min(pi, abs(pks0(:,tps(i)+1:end,:) - pks0(:,1:end-tps(i),:)));
    diffLag = abs(pks0(:,tps(i)+1:end,:) - pks0(:,1:end-tps(i),:));
    diffLagSel = diffLag(diffLag<3);
    for j = 1:length(quantiles)
        probs(i,j) = sum(diffLagSel(:) > quantiles(j))/length(diffLagSel(:));
    end
end


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
xlabel('$\Delta t$','Interpreter','latex','FontSize',20)
ylabel({'Fraction of', 'peak moved'},'FontSize',20)
set(gca,'LineWidth',1,'FontSize',16)
% ylim([0.0,0.6])
xlim([0,5e3])

%%
% ============================================================
% Analyze the fraction of neurons that change the tunning property: from
% left to right
% ============================================================
pkThreshold = 0.05;
Yt = output.Yt;
L = size(Yt,2);
tuningLR = nan(param.NE,time_points);
peakInx = nan(param.NE,time_points);

for i = 1:time_points
    [pkVal, peakPosi] = sort(Yt(:,:,i),2,'descend');
    actInx = pkVal(:,1)>pkThreshold;
    peakInx(actInx,i) = peakPosi(actInx,1);
end

% statistics of the tuning change
timedT  = round(time_points/2); % maximum seperation of time
consist = nan(timedT,2);  % mean and standard deviation
shiftTun = nan(timedT,2);
lossTun = nan(timedT,2);
gainTun = nan(timedT,2);
tuningInfo = nan(param.NE,time_points);
tuningInfo(peakInx <= round(L/2)) = -1;  % left tuning
tuningInfo(peakInx > round(L/2)) = 1;    % right tuning

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
set(gca,'FontSize',16,'LineWidth',1)

% prefix = 'Tmaze_switch_tuning';
% saveas(shitTunFig,[sFolder,filesep,prefix,'.fig'])
% print('-depsc',[saveFolder,filesep,prefix,'.eps'])


%% Save the data
% save('./data/revision/Tmaze_EI_06242022.mat','-v7.3')
