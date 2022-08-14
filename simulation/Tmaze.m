% this program test how the receptive field change in the T-maze task
% this program generates data for figure 6

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

k =300;       % number of neurons
% m = 20;     % number of interneurons

% default is a ring
tau = 0.5;
learnType = 'snsm';    % snsm if using simple non-negative similarity matching
BatchSize = 1;         % default 50

radius = 1;
t = 1e3;    % total number of samples previously 1e4

% generate the data
n = 3;     % input dimensionality, 3 for Tmaze and 2 for ring
sep = 2*pi/t;
X1 = [radius*cos(0:sep:pi-sep);radius*sin(0:sep:pi-sep)]; 
X = [X1,X1;ones(1,t/2),-ones(1,t/2)];


% plot input similarity matrix
SMX = X'*X;

figure
ax = axes;
imagesc(ax,SMX);
colorbar
ax.XTick = [1 5e3 1e4];
ax.YTick = [1 5e3 1e4];
ax.XTickLabel = {'0', '\pi', '\pi'};
ax.YTickLabel = {'0', '\pi', '\pi'};
xlabel('position','FontSize',28)
ylabel('position','FontSize',28)
set(gca,'FontSize',24)

figure
plot(X(1,:),X(2,:))

%% setup the learning parameters
noiseStd = 4e-5;      % 0.005 for ring, 1e-3 for Tmaze
learnRate = 0.01;   % default 0.05

Cx = X*X'/t;  % input covariance matrix

% store the perturbations
record_step = 100;

% initialize the states
y0 = zeros(k,BatchSize);
% estimate error
validBatch = 100; % randomly select 100 to estimate the error
Y0val = zeros(k,validBatch);

% Use offline learning to find the inital solution
params.W = 0.1*randn(k,n);
params.M = eye(k);      % lateral connection if using simple nsm
params.lbd1 = 2e-5;     % regularization for the simple nsm, 1e-3
params.lbd2 = 1e-3;        %default 1e-3

params.alpha = 1.5;  % should be smaller than 1 if for the ring model
params.beta = 1; 
params.gy = 0.05;   % update step for y
params.gz = 0.05;   % update step for z
params.gv = 0.1;   % update step for V
params.b = zeros(k,1);  % biase
params.learnRate = learnRate;  % learning rate for W and b
params.noise =  noiseStd;   % stanard deivation of noise 


if strcmp(learnType, 'snsm')
    for i = 1:1e4
        inx = randperm(t,BatchSize);
        x = X(:,inx);  % randomly select one input
%         states = MantHelper.nsmDynBatch(x,y0, params);
        states = PlaceCellhelper.nsmDynBatch(x,y0, params);
        
%         [states, params] = MantHelper.nsmDynAutopase(x,y0, params);
        y = states.Y;

        % update weight matrix
%         params.W = (1-params.learnRate)*params.W + params.learnRate*y*x'/BatchSize;
        params.W = (1-params.learnRate)*params.W + params.learnRate*y*x'/BatchSize...
            +sqrt(params.learnRate)*params.noise*randn(k,n);        
%         params.M = (1-params.learnRate)*params.M + params.learnRate*y*y'/BatchSize;
        params.M = max(0,(1-params.learnRate)*params.M + params.learnRate*y*y'/BatchSize + ...
            sqrt(params.learnRate)*params.noise*randn(k,k));
        params.b = (1-params.learnRate)*params.b + params.learnRate*sqrt(params.alpha)*mean(y,2);
    end
end

% check the receptive field
Xsel = X(:,1:5:end);
Y0 = zeros(k,size(Xsel,2));
% Z0 = zeros(m,size(Xsel,2));
% states_fixed_nn= MantHelper.nsmDynBatch(Xsel,Y0, params);
states_fixed_nn = PlaceCellhelper.nsmDynBatch(Xsel,Y0, params);
% [states_fixed_nn, params] = MantHelper.neuralDynBatch(Xsel,Y0,Z0,V, params);

Z = norm(Xsel'*Xsel - states_fixed_nn.Y'*states_fixed_nn.Y - params.alpha);

% ============ visualize the input ==============
colors = brewermap(round(t/10),'Spectral');
figure
hold on
for i = 1:round(t/10)
    plot(X(1,i*10),X(2,i*10),'.','Color',colors(i,:),'MarkerSize',10)
end
hold off
pbaspect([1 1 1])
xlabel('x','FontSize',28)
ylabel('y','FontSize',28)


% =============place field of neurons ==========
figure
imagesc(states_fixed_nn.Y)
colorbar
xlabel('position','FontSize',28)
ylabel('neuron','FontSize',28)
ax = gca;
ax.XTick = [1 500 1000];
ax.XTickLabel = {'0', '\pi', '2\pi'};
set(gca,'FontSize',24)

% =============reorder the place field of neurons ==========
% % sort the location
% [sortVal,sortedInx] = sort(states_fixed_nn.Y,2,'descend');
% [~,neurOrder] = sort(sortedInx(:,1));

% sort the neurons based on their centroid of RFs
[pkCM, ~] = PlaceCellhelper.centerMassPks1D(states_fixed_nn.Y,0.05);
actiInx = find(~isnan(pkCM));
actiPkCM = pkCM(actiInx);
[~,neurOrder] = sort(actiPkCM);



figure
imagesc(states_fixed_nn.Y(actiInx(neurOrder),:))
colorbar
xlabel('position','FontSize',28)
ylabel('neuron','FontSize',28)
ax = gca;
ax.XTick = [1 500 1000];
ax.XTickLabel = {'0', '\pi', '2\pi'};
set(gca,'FontSize',24)

% % =========== peak amplitude distribution ================
% figure
% histogram(sortVal(:,1),30)

% =========== Example place field ========================
% sort and find the
num_acti = length(actiInx);
sel_inx = actiInx(neurOrder(1:round(num_acti/7):num_acti));
rfExamples = states_fixed_nn.Y(sel_inx,:);
figure
for i = 1:length(sel_inx)
    plot(states_fixed_nn.Y(sel_inx(i),:),'Color',set1(i,:),'LineWidth',3)
    hold on
end
hold off
xlabel('position','FontSize',28)
ylabel('response','FontSize',28)
ax = gca;
ax.XTick = [1 500 1000];
ax.XTickLabel = {'0', '\pi', '2\pi'};
set(gca,'FontSize',24)

%============= similarity matrix of ordered represenation =================
SM = states_fixed_nn.Y(actiInx(neurOrder),:)'*states_fixed_nn.Y(actiInx(neurOrder),:);
figure
imagesc(SM)
colorbar
ax = gca;
ax.XTick = [1 500 1000];
ax.YTick = [1 500 1000];
ax.XTickLabel = {'0', '\pi', '2\pi'};
ax.YTickLabel = {'0', '\pi', '2\pi'};
xlabel('position','FontSize',28)
ylabel('position','FontSize',28)
set(gca,'FontSize',24)



%% continue updating with noise

tot_iter = 2e4;
num_sel = 200;
step = 50;
time_points = round(tot_iter/step);

ystore = nan(tot_iter,1);
% store the weight matrices to see if they are stable
allW = nan(k,n,time_points);
allM = nan(k,k,time_points);
allbias = nan(k,time_points);

% testing data, only used when check the representations
Xsel = X(:,1:4:end);   % modified on 7/18/2020
Y0 = zeros(k,size(Xsel,2));
Yt = nan(k,size(Xsel,2),time_points);

if strcmp(learnType, 'snsm')
    for i = 1:tot_iter
        inx = randperm(t,BatchSize);
        x = X(:,inx);  % randomly select one input
%         states= MantHelper.nsmDynBatch(x,y0,params);
        states = PlaceCellhelper.nsmDynBatch(x,y0, params);
%         [states, params] = MantHelper.nsmNoiseDynBatch(x,y0,params);
        y = states.Y;
%         ystore(i) = y;
        
        % update weight matrix   
        params.W = (1-params.learnRate)*params.W + params.learnRate*y*x'/BatchSize...
            +sqrt(params.learnRate)*params.noise*randn(k,n);        
        params.M = max(0,(1-params.learnRate)*params.M + params.learnRate*y*y'/BatchSize....
            +sqrt(params.learnRate)*params.noise*randn(k,k));
        params.b = (1-params.learnRate)*params.b + params.learnRate*sqrt(params.alpha)*mean(y,2);
 
        % store and check representations
        if mod(i, step) == 0
%             [states_fixed, params] = MantHelper.nsmDynBatch(Xsel,Y0,params);
            states_fixed = PlaceCellhelper.nsmDynBatch(Xsel,Y0, params);
            Yt(:,:,round(i/step)) = states_fixed.Y;
%             for samp = 1:size(Xsel,2)
%                 states_fixed = PlaceCellhelper.nsmDynBatch(x,y0, params);
%                 Yt(:,samp,round(i/step)) = states_fixed.Y;
%             end
        end
    end 
end

%% Analysis, check the change of place field
pkThreshold = 0.05;  % active threshold


% peak of receptive field
peakInx = nan(k,time_points);
peakVal = nan(k,time_points);
peakPosi = nan(k,time_points);
for i = 1:time_points
%     [pkVal, peakPosi] = sort(Yt(:,:,i),2,'descend');
%     peakInx(:,i) = peakPosi(:,1);
%     peakVal(:,i) = pkVal(:,1);
    
    % based on centroid of RFs
    [pkCM, ~] = PlaceCellhelper.centerMassPks1D(Yt(:,:,i),0.05);
%     actiInx = find(~isnan(pkCM));
%     actiPkCM = pkCM(actiInx);
%     [~,neurOrder] = sort(actiPkCM);
    peakPosi(:,i) = pkCM;
end

% ======== faction of neurons have receptive field at a give time =====

% quantified by the peak value larger than a threshold 0.01
% rfIndex = peakVal > pkThreshold;

% fraction of neurons
activeRatio = mean(~isnan(peakPosi),1);
figure
plot(1:time_points,activeRatio(1:end))
xlabel('iteration')
ylabel('active fraction')
ylim([0,1])

figure
histogram(activeRatio(101:end))

% drop in 
% figure
% plot(sum(rfIndex,2)/k)


% =========place field order by the begining ======================
inxSel = [100, 200, 400];
figure
for i = 1:length(inxSel)
    subplot(1,3,i)
    imagesc(Yt(:,:,inxSel(i)))
    colorbar
    title(['iteration ', num2str(inxSel(i))])
    ax = gca;
    ax.XTick = [1 500 1000];
    ax.XTickLabel = {'0', '\pi', '2\pi'};
    ylabel('neuron index')
    xlabel('position')
end


% ======== ordered by current index ==========
figure
for i = 1:length(inxSel)
%     actiInx = find(~isnan(pkCM));
%     actiPkCM = pkCM(actiInx);
    [~,neuroInx] = sort(peakPosi(:,inxSel(i)));

%     [~,neuroInx] = sort(peakInx(:,inxSel(i)));
    subplot(1,3,i)
    imagesc(Yt(neuroInx,:,inxSel(i)))
    colorbar
    ax = gca;
    ax.XTick = [1 500 1000];
    ax.XTickLabel = {'0', '\pi', '2\pi'};
    title(['iteration ', num2str(inxSel(i))])
    xlabel('position')
    ylabel('sorted index')
end

% ordered by time point 1
inxSelPV = [1,100,300];
actiInx = find(~isnan(peakPosi(:,inxSelPV(1))));
actiPkCM = peakPosi(actiInx,inxSelPV(1));
[~,inx] = sort(actiPkCM);
neuroInx = actiInx(inx);   % only use neurons that are active at time 1
sepInx = find(peakPosi(neuroInx,inxSelPV(1))>size(Yt,2)/2,1,'first');


YtMerge = cat(1,Yt(neuroInx(1:sepInx-1),1:size(Yt,2)/2,:),Yt(neuroInx(sepInx:end),size(Yt,2)/2+1:end,:));
MergeaPkCM = [peakPosi(neuroInx(1:sepInx-1),inxSelPV(1));peakPosi(neuroInx(sepInx:end),inxSelPV(1))-size(Yt,2)/2];
[~, newInx] = sort(MergeaPkCM,'ascend');


figure
for i = 1:length(inxSelPV)
    subplot(1,3,i)
    imagesc(YtMerge(newInx,:,inxSelPV(i)),[0,0.5])
    colorbar
    ax = gca;
%     ax.XTick = [1 500 1000];
    ax.XTickLabel = {'0', '\pi', '2\pi'};
    title(['iteration ', num2str(inxSelPV(i))])
    xlabel('position')
    ylabel('sorted index')
end


% ======== representation similarity matrix =======
figure
for i = 1:length(inxSel)
%     [~,neuroInx] = sort(peakInx(:,inxSel(i)));
    SM = Yt(:,:,inxSel(i))'*Yt(:,:,inxSel(i));
    subplot(1,3,i)
    imagesc(SM,[0,0.6])
    colorbar
    ax = gca;
    ax.XTick = [1 500 1000];
    ax.YTick = [1 500 1000];
    ax.XTickLabel = {'0', '\pi', '2\pi'};
    ax.YTickLabel = {'0', '\pi', '2\pi'};
    title(['iteration', num2str(inxSel(i))])
    xlabel('position')
    ylabel('position')
end


%% shift of centroids

adjPeaks = mod(peakPosi, size(Yt,2)/2);
psRange = pi;   % the range of ring positions
pks0 = adjPeaks/size(Yt,2)*2*psRange;
% tps = [1,2,3,8];
tps = [50,100,150,200];
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
    plot(tps'*step,probs(:,i),'o-','MarkerSize',8,'Color',blues(1+3*i,:),...
        'MarkerFaceColor',blues(1+3*i,:),'MarkerEdgeColor',blues(1+3*i,:),'LineWidth',1.5)
end
hold off
box on

% lg = legend('\Delta s >1/4 L','\Delta s>1/2 L','\Delta s >3/4 L');
lg = legend('\Delta s > 0.07 L','\Delta s > 0.11 L','\Delta s > 0.22 L');
set(lg,'FontSize',12)
xlabel('$\Delta t$','Interpreter','latex','FontSize',20)
ylabel({'Fraction of', 'peak moved'},'FontSize',20)
set(gca,'LineWidth',1,'FontSize',16)
ylim([0.2,0.6])
xlim([0,450])

%%
% ============================================================
% Analyze the fraction of neurons that change the tunning property: from
% left to right
% ============================================================
L = size(Yt,2);
tuningLR = nan(k,time_points);
peakInx = nan(k,time_points);
% peakInxLeft = nan(k,time_points);
% peakValLeft = nan(k,time_points);
% peakInxRight = nan(k,time_points);
% peakValRight = nan(k,time_points);
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
tuningInfo = nan(k,time_points);
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

% prefix = 'Tmaze_switch_tuning';
% saveas(shitTunFig,[sFolder,filesep,prefix,'.fig'])
% print('-depsc',[saveFolder,filesep,prefix,'.eps'])

%% save data

