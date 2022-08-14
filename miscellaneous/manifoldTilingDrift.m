% this program test how the receptive field change in the manifold tiling
% paper

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

k =200;     % number of neurons
n = 2;      % input dimensionality, 3 for Tmaze and 2 for ring
m = 20;     % number of interneurons

% default is a ring
dataType = 'Tmaze'; % 'Tmaze' or 'ring';
tau = 0.5;
learnType = 'batch'; % snsm if using simple non-negative similarity matching
BatchSize = 20;    % default 50


radius = 1;
t = 1e3;    % total number of samples previously 1e4
if strcmp(dataType, 'ring')
    n = 2;     % input dimensionality, 3 for Tmaze and 2 for ring
    sep = 2*pi/t;
    ags =0:sep:2*pi;
    X = [radius*cos(0:sep:2*pi);radius*sin(0:sep:2*pi)];

elseif strcmp(dataType, 'semi-ring')
    n = 2;     % input dimensionality, 3 for Tmaze and 2 for ring
    sep = pi/t;
    X = [radius*cos(0:sep:pi);radius*sin(0:sep:pi)];
elseif strcmp(dataType, 'Tmaze')
    n = 3;     % input dimensionality, 3 for Tmaze and 2 for ring
    scale = 0.5; % the context encoding
    sep = 2*pi/t;
    X1 = [radius*cos(0:sep:pi-sep);radius*sin(0:sep:pi-sep)]; 
    X = [X1,X1;scale*ones(1,t/2),-ones(1,t/2)*scale];

end
% plot(X(1,:),X(2,:),'.')

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
noiseStd = 1e-3;      % 0.005 for ring, 1e-3 for Tmaze
learnRate = 0.05;   % default 0.05

Cx = X*X'/t;  % input covariance matrix

% store the perturbations
record_step = 100;

% initialize the states
y0 = zeros(k,BatchSize);
z0 = zeros(m,BatchSize);
V = rand(m,k);  % feedback from cortical neurons
% Wout = zeros(1,k);      % linear decoder weight vector

% estimate error
validBatch = 100; % randomly select 100 to estimate the error
Y0val = zeros(k,validBatch);

% Use offline learning to find the inital solution
params.W = 0.1*randn(k,n);
% params.W = zeros(k,n);
params.M = eye(k);      % lateral connection if using simple nsm
params.lbd1 = 1e-4;     % regularization for the simple nsm, 1e-3
params.lbd2 = 1e-3;        %default 1e-3

params.alpha = 0;  % should be smaller than 1 if for the ring model
params.beta = 1; 
params.gy = 0.05;   % update step for y
params.gz = 0.05;   % update step for z
params.gv = 0.1;   % update step for V
params.b = zeros(k,1);  % biase
params.learnRate = learnRate;  % learning rate for W and b
params.noise =  noiseStd;   % stanard deivation of noise 
params.read_lr = learnRate*5;     % learning rate of readout matrix

% params.currNosie = 0.05;    % current noise
readError = [];   % store the readout error of a linear decoder

% store the W and bias
% tot = 5e3;
% step = 10;
% allW = nan(round(tot/step),n);
% allbias = nan(round(tot/step),1);
% noise1 = noiseStd;
% noise2 = 0;
if strcmp(learnType, 'online')
    for i = 1:1e4
        x = X(:,randperm(t,1));  % randomly select one input
        [states, params] = MantHelper.neuralDynOnline(x,y0,z0,V, params);
        y = states.y;
        % update weight matrix
        params.W = (1-params.learnRate)*params.W + y*x';
        params.b = (1-params.learnRate)*params.b + params.learnRate*sqrt(params.alpha)*y;
        V = states.V;   % current 
    end
elseif strcmp(learnType, 'batch')
    for i = 1:2e3
        x = X(:,randperm(t,BatchSize));  % randomly select one input
        [states, params] = MantHelper.neuralDynBatch(x,y0,z0,V, params);
        y = states.Y;
        % update weight matrix
        params.W = (1-params.learnRate)*params.W + params.learnRate*y*x'/BatchSize;
        params.b = (1-params.learnRate)*params.b + params.learnRate*sqrt(params.alpha)*mean(y,2);
        V = states.V;   % current 
    end
elseif strcmp(learnType, 'snsm')
    for i = 1:1e3
        inx = randperm(t,BatchSize);
        x = X(:,inx);  % randomly select one input
        [states, params] = MantHelper.nsmDynBatch(x,y0, params);
%         [states, params] = MantHelper.nsmDynAutopase(x,y0, params);
        y = states.Y;

        % update weight matrix
%         params.W = (1-params.learnRate)*params.W + params.learnRate*y*x'/BatchSize;
        params.W = (1-params.learnRate)*params.W + params.learnRate*y*x'/BatchSize...
            +sqrt(params.learnRate)*params.noise*randn(k,n);        
        params.M = (1-params.learnRate)*params.M + params.learnRate*y*y'/BatchSize;
%         params.M = max(0,(1-params.learnRate)*params.M + params.learnRate*y*y'/BatchSize + ...
%             sqrt(params.learnRate)*params.noise*randn(k,k));
        params.b = (1-params.learnRate)*params.b + params.learnRate*sqrt(params.alpha)*mean(y,2);
%         V = states.V;   % current 

        % Assume a linear readout
        
%         label = inx/t*2*pi - pi;  % from -pi to pi
%         Wout = Wout + params.learnRate*(label - Wout*y)*y'/BatchSize;
        
%         if mod(i,100) == 1
%             valSel = randperm(t,validBatch);
%             valLabels = valSel/t*2*pi - pi;
%             [states, ~] = MantHelper.nsmDynBatch(X(:,valSel),Y0val, params);
%             readError = [readError, mean((valLabels - Wout*states.Y).^2)];
%         end
    end
end

% check the receptive field
Xsel = X(:,1:5:end);
Y0 = zeros(k,size(Xsel,2));
Z0 = zeros(m,size(Xsel,2));
% [states_fixed_nn, params] = MantHelper.nsmDynBatch(Xsel,Y0, params);
[states_fixed_nn, params] = MantHelper.neuralDynBatch(Xsel,Y0,Z0,V, params);

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
% sort the location
[sortVal,sortedInx] = sort(states_fixed_nn.Y,2,'descend');
[~,neurOrder] = sort(sortedInx(:,1));
figure
imagesc(states_fixed_nn.Y(neurOrder,:))
colorbar
xlabel('position','FontSize',28)
ylabel('neuron','FontSize',28)
ax = gca;
ax.XTick = [1 500 1000];
ax.XTickLabel = {'0', '\pi', '2\pi'};
set(gca,'FontSize',24)

% =========== peak amplitude distribution ================
figure
histogram(sortVal(:,1),30)

% =========== Example place field ========================
% sort and find the
sel_inx = neurOrder(50:round(k/7):k);
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
SM = states_fixed_nn.Y(neurOrder,:)'*states_fixed_nn.Y(neurOrder,:);
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


%============== Check the readout performance ===============
valSel = randperm(t,validBatch);
valLabels = valSel/t*2*pi - pi;
[states, ~] = MantHelper.nsmDynBatch(X(:,valSel),Y0val, params);
% z = Wout*states.Y;
% 
% figure
% plot(valLabels,z,'o')
% xlabel('True position (rad)')
% ylabel('Predicted position (rad)')


% trace of error
% figure
% plot(readError)
% xlabel('Time')
% ylabel('Mean square error')

%% continue updating with noise
% BatchSize = 1;  % this used only during debug 
tot_iter = 1e4;
num_sel = 200;
step = 20;
time_points = round(tot_iter/step);

ystore = nan(tot_iter,1);
% store the weight matrices to see if they are stable
allW = nan(k,n,time_points);
allM = nan(k,k,time_points);
allbias = nan(k,time_points);

% testing data, only used when check the representations
Xsel = X(:,1:2:end);   % modified on 7/18/2020
Y0 = zeros(k,size(Xsel,2));
Z0 = zeros(m,size(Xsel,2));
Yt = nan(k,size(Xsel,2),time_points);
if strcmp(learnType, 'online')
    for i = 1:tot_iter
        x = X(:,randperm(t,1));  % randomly select one input
        [states, params] = MantHelper.neuralDynOnline(x,y0,z0,V, params);
        y = states.y;
        % update weight matrix
        params.W = (1-params.learnRate)*params.W + params.learnRate*y*x' + ...
            sqrt(params.learnRate)*params.noise*randn(k,2);
        params.b = (1-params.learnRate)*params.b + params.learnRate*sqrt(params.alpha)*y;
        V = states.V;   % current feedback matrix
        
    end
elseif strcmp(learnType, 'batch')
    for i = 1:tot_iter
        x = X(:,randperm(t,BatchSize));  % randomly select one input
        [states, params] = MantHelper.neuralDynBatch(x,y0,z0,V, params);
        y = states.Y;
        % update weight matrix
        params.W = (1-params.learnRate)*params.W + params.learnRate*y*x'/BatchSize + ...
            sqrt(params.learnRate)*params.noise*randn(k,n);  % adding noise
        params.b = (1-params.learnRate)*params.b + params.learnRate*sqrt(params.alpha)*mean(y,2);
        V = states.V;   % current
        
        % store and check representations
        if mod(i, step) == 0
            [states_fixed, params] = MantHelper.neuralDynBatch(Xsel,Y0,Z0,V, params);
            Yt(:,:,round(i/step)) = states_fixed.Y;
%             allW(:,:,round(i/step)) = params.W;
        end
    end
elseif strcmp(learnType, 'snsm')
    for i = 1:tot_iter
        inx = randperm(t,BatchSize);
        x = X(:,inx);  % randomly select one input
        [states, params] = MantHelper.nsmDynBatch(x,y0,params);
%         [states, params] = MantHelper.nsmNoiseDynBatch(x,y0,params);
        y = states.Y;
%         ystore(i) = y;
        
        % update weight matrix   
        params.W = (1-params.learnRate)*params.W + params.learnRate*y*x'/BatchSize...
            +sqrt(params.learnRate)*params.noise*randn(k,n);        
        params.M = max(0,(1-params.learnRate)*params.M + params.learnRate*y*y'/BatchSize....
            +sqrt(params.learnRate)*params.noise*randn(k,k));
%         params.M = (1-params.learnRate)*params.M + params.learnRate*y*y'/BatchSize;
        params.b = (1-params.learnRate)*params.b + params.learnRate*sqrt(params.alpha)*mean(y,2);
        
        % readout
%         label = inx/t*2*pi - pi;  % from -pi to pi
%         Wout = Wout + params.learnRate*(label - Wout*y)*y'/BatchSize;
        
%         if mod(i,100) == 1
%             valSel = randperm(t,validBatch);
%             valLabels = valSel/t*2*pi - pi;
%             [states, ~] = MantHelper.nsmDynBatch(X(:,valSel),Y0val, params);
%             readError = [readError, mean((valLabels - Wout*states.Y).^2)];
%         end
        
        % store and check representations
        if mod(i, step) == 0
            [states_fixed, params] = MantHelper.nsmDynBatch(Xsel,Y0,params);
            Yt(:,:,round(i/step)) = states_fixed.Y;
%             allW(:,:,round(i/step)) = params.W;
%             allM(:,:,round(i/step)) = params.M;
%             allbias(:,round(i/step)) = params.b;
        end
    end 
end

%% Analysis of one neuron case
% all Ys
% allY = squeeze(Yt(100,:,:));
allY = squeeze(Yt(1,:,:));

figure
imagesc(allY,[0,0.2])
xlabel('time')
ylabel('position index')
colorbar


% the weight matrix
ws = squeeze(allW(1,:,:));
figure
plot(ws(1,:)',ws(2,:)','.')
xlabel('$w_1$', 'Interpreter','latex')
ylabel('$w_2$', 'Interpreter','latex')


figure
plot((1:size(ws,2))',ws','LineWidth',1.5)
xlabel('time')
ylabel('$w_{1,2}$','Interpreter','latex')


% norm of the weight vector
wnorm = sqrt(sum(ws.^2,1));
figure
plot(sqrt(sum(ws.^2,1)))

figure
histogram(wnorm)
xlabel('$||w||_2$','Interpreter','latex')
ylabel('count')



%% Analysis, check the change of place field
pkThreshold = 0.02;  % active threshold


% peak of receptive field
peakInx = nan(k,time_points);
peakVal = nan(k,time_points);
for i = 1:time_points
    [pkVal, peakPosi] = sort(Yt(:,:,i),2,'descend');
    peakInx(:,i) = peakPosi(:,1);
    peakVal(:,i) = pkVal(:,1);
end

% ======== faction of neurons have receptive field at a give time =====
% quantified by the peak value larger than a threshold 0.01
rfIndex = peakVal > pkThreshold;

% fraction of neurons
activeRatio = sum(rfIndex,1)/k;
figure
plot(101:time_points,activeRatio(101:end))
xlabel('iteration')
ylabel('active fraction')

figure
histogram(activeRatio(101:end))

% drop in 
figure
plot(sum(rfIndex,2)/k)


% ===========

% =========place field order by the begining ======================
inxSel = [100, 250, 500];
% inxSel = [100,150,200];
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
    [~,neuroInx] = sort(peakInx(:,inxSel(i)));
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
inxSelPV = [200,300,400];
[~,neuroInx] = sort(peakInx(:,inxSelPV(1)));

% merged left and right
YtMerge = cat(1,Yt(neuroInx(1:round(k/2)),1:500,:),Yt(neuroInx((k/2+1):end),501:1000,:));
% YtMerge = cat(1,Yt(neuroInx(1:round(k/2)),1:100,:),Yt(neuroInx((k/2+1):end),101:200,:));

mergePeakInx = nan(k,time_points);
for i = 1:time_points
    [~,mergePeakPosi] = sort(YtMerge(:,:,i),2,'descend');
    mergePeakInx(:,i) = mergePeakPosi(:,1);
end
[~, newInx] = sort(mergePeakInx(:,inxSelPV(1)));

figure
for i = 1:length(inxSelPV)
    subplot(1,3,i)
    imagesc(YtMerge(newInx(100:k),:,inxSelPV(i)),[0,0.4])
    colorbar
    ax = gca;
    ax.XTick = [1 500 1000];
    ax.XTickLabel = {'0', '\pi', '2\pi'};
    title(['iteration ', num2str(inxSelPV(i))])
    xlabel('position')
    ylabel('sorted index')
end


% ======== place field of a single neurons across repeats =============
figure
epLoc = [1, 100, 150];
for i = 1:3
    ypart = squeeze(Yt(epLoc(i),1:500,:));
    subplot(1,3,i)
    imagesc(ypart')
    colorbar
    xlabel('location')
    ylabel('time')
%     set(gca,'XTick',100:100:500,'XTickLabel',['0.1','0.2','0.3','0.4','0.5'])
    set(gca,'XTick',[0, 250, 500],'XTickLabel',{'0','0.5','1'})
    title(['neuron ',num2str(epLoc(i))])
end


% ======== representation similarity matrix =======
figure
for i = 1:length(inxSel)
%     [~,neuroInx] = sort(peakInx(:,inxSel(i)));
    SM = Yt(:,:,inxSel(i))'*Yt(:,:,inxSel(i));
    subplot(1,3,i)
    imagesc(SM,[0,3])
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

% auto correlation function with the first snapshot
refTime = 100;  % reference iterations
acfSM = nan(size(Yt,3) - refTime,1);
% acfSM = nan(size(Yt,3),1);
% acfCoef = nan(size(Yt,3),1);
acfCoef = nan(size(Yt,3) - refTime,1);
refY = Yt(:,:,100);  % reference
vecRef = refY(:);
for i = refTime:(size(Yt,3)-1)
    cmpY = Yt(:,:,i);   
    acfSM(i-refTime+1) = vecRef'*cmpY(:)/size(Yt,2);
    cm = corrcoef(vecRef,cmpY(:));
    acfCoef(i-refTime+1) = cm(1,2);
end

% fit an exponential decaying curve
xFit = (1:200)';
yFit = acfSM(xFit);
fexp1 = fit(xFit,yFit,'exp1');
fexp2 = fit(xFit,acfCoef(xFit),'exp1');

modelfun = @(b,x)(b(1)+b(2)*exp(-b(3)*x));
opts = statset('nlinfit');
opts.RobustWgtFun = 'bisquare';
beta1 = nlinfit(xFit,yFit,modelfun,[0.5,2,1e-2],opts);
beta2 = nlinfit(xFit,acfCoef(xFit),modelfun,[0,1,1e-2],opts);


figure
title('Comparison with first snapshot')
subplot(1,2,1)
plot(acfSM,'LineWidth',3)
hold on
plot(xFit,modelfun(beta1,xFit),'LineWidth',3)
hold off
xlabel('iteration','FontSize',24)
ylabel('similarity','FontSize',24)
set(gca,'LineWidth',1.5,'FontSize',20)

subplot(1,2,2)
plot(acfCoef,'LineWidth',3)
hold on
plot(xFit,modelfun(beta2,xFit),'LineWidth',3)
xlabel('iteration','FontSize',24)
ylabel('corr. coef','FontSize',24)
set(gca,'LineWidth',1.5,'FontSize',20)


% ============ Simple linear decoder ===============================
% train a linear decoder at each "day" and test it on other trial
% in the same day. In the model, with-in day trials can be thought as the
% trials that are close to the trianing trials in time

% trian the classifer base on vector at iteration 100
timeSel = 200;          % select a time point to train the decoder
popVec = Yt(:,1:1000,timeSel);  % a matrix
xloc = (1:1000)/500-1;    %position information

% Wout = MantHelper.fitWout([popVec,Yt(:,1:1000,200)],[xloc,xloc]);
lbd = 0.1;
wout = xloc*popVec'*pinv(popVec*popVec' + lbd*eye(size(popVec,1)));
% Wout = MantHelper.fitWoutReg(popVec,xloc,0.005, 0.1);
figure
% plot(xloc,tanh(wout*popVec),'LineWidth',3)
plot(xloc,wout*popVec,'LineWidth',3)
hold on
plot([-1;1],[-1;1],'--')
hold off
xlabel('Real position')
ylabel('Predicted position')
set(gca,'LineWidth',1.5,'FontSize',20)

% prediction on subsequent time point
dT = 100;
allErr = nan(dT,1);

for i = 1:dT
    xpred = wout*Yt(:,1:1000,timeSel+i);
%     xpred = tanh(Wout*Yt(:,1:1000,timeSel+i));
    allErr(i) = mean(abs(xpred - xloc));   % absolute average decoding error
end

% plot the time dependent prediction error
figure
plot((1:dT),allErr/2,'LineWidth',3)
xlabel('$\Delta t$', 'Interpreter','latex')
ylabel('Decoding error: $|\Delta x|/L$', 'Interpreter','latex')
set(gca,'LineWidth',1.5,'FontSize',20)

% prediction vs real
figure
plot(xloc,xpred,'LineWidth',3)
hold on
plot([-1;1],[-1;1],'--')
hold off
xlabel('Real position')
ylabel('Predicted position')
set(gca,'LineWidth',1.5,'FontSize',20)


% test
[~,sortix] = sort(popVec,2,'descend');
[~,nInx] = sort(sortix(:,1));
newPop = popVec(nInx,:);
figure
imagesc(newPop)


% ========== Quantify the drifting behavior ==========================
% due to the periodic condition manifold, we need to correct the movement
% using the initial order as reference point
orderedPeaks = peakInx(neurOrder,:);
shift = diff(orderedPeaks')';
reCode = zeros(size(shift));
reCode(shift > 550) = -1;
reCode(shift < -550) = 1;
addVals = cumsum(reCode,2)*2*pi;
newPeaks = orderedPeaks/1001*2*pi + [zeros(k,1),addVals];
% newPeaks = orderedPeaks/1001*2*pi;   % 11/21/2020


% check the shift
figure
plot(newPeaks(1:5,:)')
xlabel('iteration','FontSize',28)
ylabel('shift of RF (rad)','FontSize',28)
ax =  gca;
% ax.YTick = (-3:1:3)*pi;
% ax.YTickLabel = {'-3\pi','-2\pi', '-\pi','0', '\pi', '2\pi','3\pi'};
set(gca,'FontSize',24)


% plot the average shift with respect to time
% based on original positions
msds0= nan(floor(time_points/2),k);
pks0 = orderedPeaks/1001*2*pi;
for i = 100:floor(time_points/2)
    diffLag = min(pi, abs(pks0(:,i+1:end,:) - pks0(:,1:end-i,:)));
    msds0(i,:) = mean(diffLag,2);
end

% selective plot the distance distribution 
tpoints = [1,10,50,100,400];
figure
hold on
for i = 1:length(tpoints)
    diffLag = min(pi, abs(pks0(:,tpoints(i)+1:end,:) - pks0(:,1:end-tpoints(i),:)));
    [F,ds_x] = ecdf(diffLag(:));
    plot(ds_x/pi,F)
%     histogram(diffLag(:))
    
end
hold off
box on
lg = legend('t = 4', 't =40', 't =200','t = 400', 't = 1600','Location','southeast');
set(lg,'FontSize',16)
xlabel('Peak shift (L)','FontSize',24)
ylabel('Cumulative distribution','FontSize',24)
set(gca,'FontSize',24,'LineWidth',1.5)


% another way to show the results
tps = [1,5,10,20,50,100,150,200];
quantiles = (1:3)/10*2*pi;
% quantiles = (1:3)/4*pi;

probs = nan(length(tps),length(quantiles));
for i = 1:length(tps)
    diffLag = min(pi, abs(pks0(:,tps(i)+1:end,:) - pks0(:,1:end-tps(i),:)));
    for j = 1:length(quantiles)
        probs(i,j) = sum(diffLag(:)>quantiles(j))/length(diffLag(:));
    end
end
figure
plot(tps'*step,probs,'o-','MarkerSize',10)
legend('>0.1L','>0.2L','>0.3L')
xlabel('$\Delta t$','Interpreter','latex','FontSize',24)
ylabel('Probability','FontSize',24)
set(gca,'LineWidth',1.5,'FontSize',24)




% estimate the diffusion constant
msds = nan(floor(time_points/2),k);
for i = 1:floor(time_points/2)
    diffLag = newPeaks(:,i+1:end,:) - newPeaks(:,1:end-i,:);
%     diffLag = min(abs(newPeaks(:,i+1:end,:) - newPeaks(:,1:end-i,:)),...
%         2*pi - abs(newPeaks(:,i+1:end,:) - newPeaks(:,1:end-i,:)) );
    msds(i,:) = mean(diffLag.*diffLag,2);
end

figure
plot(msds)
xlabel('time')
ylabel('$\langle(\Delta \theta)^2\rangle$','Interpreter','latex')

diffEucl = nan(k,2);  % store the diffusion constants and the exponents
for i = 1:k
    [diffEucl(i,1),diffEucl(i,2)] = MantHelper.fitMsd(msds(:,i),step);
end

% plot the distribution of diffusion constants and exponents
figure
subplot(1,2,1)
histogram(log(diffEucl(:,1)))
title('diffusion constant')
xlabel('$\ln(D)$','Interpreter','latex','FontSize',28)
ylabel('count','FontSize',28)
set(gca,'FontSize',24)


subplot(1,2,2)
histogram(diffEucl(:,2))
title('exponent: $D\sim t^{\gamma}$','Interpreter','latex')
xlabel('$\gamma$','Interpreter','latex','FontSize',28)
ylabel('count','Interpreter','latex','FontSize',28)
set(gca,'FontSize',24)

% [ce, ~,~,~,ep] = pca(z');
% Collective dynamics of the peak positions, using PCA to project the
% dynamics 

numPcSel = 10;
[coefPK,scorePK,~, ~, pkExpl] = pca(newPeaks');
projPkDyn = coefPK(:,1:numPcSel)'*newPeaks;


% numPcSel = 5;
% inx = randperm(200,100);
% [cfPk,scPK,~, ~, pkVar] = pca(pks0(inx,:)');
% pjDyn = cfPk(:,1:numPcSel)'*pks0(inx,:);
% plot(pjDyn')
% 
% % check the structure of the peak position
% Cpk = newPeaks'*newPeaks;
% figure
% imagesc(Cpk)
% colorbar
% xlabel('time','FontSize',24,'LineWidth',3)
% ylabel('time','FontSize',24,'LineWidth',3)

% figure
% plot(diag(Cpk),'LineWidth',3)
% xlabel('time','FontSize',24,'LineWidth',3)
% ylabel('variance','FontSize',24,'LineWidth',3)
% set(gca,'LineWidth',1.5,'FontSize',20)

%fitting the loading amplitude with a linear regression
% average peak and bottom value
loadAmp = max(abs(projPkDyn),[],2);
logY = log(loadAmp);
logX = [ones(numPcSel,1),log(pkExpl(1:numPcSel))];
b = logX\logY;

figure
plot(pkExpl(1:numPcSel),loadAmp,'o','MarkerSize',10,'MarkerFaceColor',blues(9,:),...
    'MarkerEdgeColor',blues(9,:))
hold on
plot(pkExpl(1:numPcSel),exp(logX*b),'LineWidth',2,'Color',rb(2,:))
set(gca,'XScale','log','YScale','log')
xlabel('Explained variance')
ylabel('Maxium amplitude')

% another plot
figure
plot(sqrt(pkExpl(1:numPcSel)),loadAmp,'o-','MarkerSize',10,'MarkerFaceColor',blues(9,:),...x
    'MarkerEdgeColor',blues(9,:),'Color',rb(2,:))
xlabel('(Explained variance)^{1/2}')
ylabel('Maxium amplitude')

% loading amplitude decays with rank of pcs
logY = log(loadAmp);
logX = [ones(numPcSel,1),log((1:numPcSel)')];
b = logX\logY;
figure
plot(loadAmp,'o','MarkerSize',10,'MarkerFaceColor',blues(9,:),...
    'MarkerEdgeColor',blues(9,:))
hold on
plot(exp(logX*b),'LineWidth',2,'Color',rb(2,:))
hold off
xlabel('PC rank')
ylabel('maximum amplitude')
set(gca,'XScale','log','YScale','log')


% distribution of pc loading, fit a power law
logY = log(pkExpl(1:numPcSel));
logX = [ones(10,1),log((1:numPcSel)')];
b = logX\logY;
figure
plot(pkExpl(1:numPcSel),'o','MarkerSize',10,'MarkerFaceColor',blues(9,:),...
    'MarkerEdgeColor',blues(9,:))
hold on
plot(exp(logX*b),'LineWidth',2,'Color',rb(2,:))
hold off
xlabel('PC rank')
ylabel('Variance explained')
set(gca,'XScale','log','YScale','log')



figure
plot((1:size(projPkDyn,2))',projPkDyn','LineWidth',3)
lg = legend('pc 1', 'pc 2', 'pc 3', 'pc 4', 'pc 5');
legend boxoff
xlabel('iteration','FontSize',24)
ylabel('pc','FontSize',24)
set(gca,'LineWidth',1.5, 'FontSize',20)


colors = brewermap(size(Yt,3),'Spectral');
figure
hold on
for i = 1:size(projPkDyn,2)
    plot3(projPkDyn(1,i),projPkDyn(2,i),projPkDyn(3,i),'.','Color',colors(i,:),'MarkerSize',15)
end
hold off
grid on
xlabel(['pc1,',num2str(round(pkExpl(1)*100)/100),'%'],'FontSize',24)
ylabel(['pc2,',num2str(round(pkExpl(2)*100)/100),'%'],'FontSize',24)
zlabel(['pc3,',num2str(round(pkExpl(3)*100)/100),'%'],'FontSize',24)
set(gca,'LineWidth',1.5, 'FontSize',20)



%% ================amplitude of peak =========================
% compare the shif of RFs
MantHelper.analyzePksTmaze(Yt,0.05)

ampls = nan(k,length(inxSel));
figure
for i = 1:length(inxSel)
    pinx = peakInx(:,inxSel(i));
    for j = 1:k
        ampls(j,i) = Yt(j,pinx(j),inxSel(i));
    end
    
    subplot(1,length(inxSel),i)
    histogram(ampls(:,i))
    xlabel('peak ampitude','FontSize',28)
    ylabel('count','Interpreter','latex','FontSize',28)
    set(gca,'FontSize',24)

end

% plot how the peak value of unit change with iteration
figure
plot(ampls)
xlabel('iteration')
ylabel('amplitude')


% Overlap, or the proability of stbale neurons across time
activeInx = peakVal > pkThreshold;
refInx = find(activeInx(:,1));  % reference
overlaps = double(activeInx(refInx,1)')*double(activeInx(refInx,:)/length(refInx));

figure
hold on
plot(1:size(peakVal,2)',overlaps,'LineWidth',3)
plot([0;50],mean(activeInx(:))*ones(2,1),'LineWidth',3)
xlim([0,50])
hold off
box on
legend('Model','Random')
xlabel('Time','FontSize',28)
ylabel('Overlap','FontSize',28)
set(gca,'LineWidth',1,'FontSize',24)

% interval length where place fields disappears
pkFlags = peakVal > pkThreshold;
silentInter = [];  % store the silent interval
activeInter = [];
randActiInter = [];  % compare with random case
actInxPerm = reshape(pkFlags(randperm(size(pkFlags,1)*size(pkFlags,2))),size(pkFlags));
for i = 1:size(peakVal,1)
    I =  find(pkFlags(i,:));
    temp = diff(I);
    silentInter = [silentInter,temp(temp>1)-1];
    
    Ia = find(pkFlags(i,:)==0);
    temp = diff(Ia);
    activeInter = [activeInter,temp(temp>1)-1];
    
    % for random permutation
    Ib = find(actInxPerm(i,:)==0);
    temp = diff(Ib);
    randActiInter = [randActiInter,temp(temp>1)-1];
end

% tabulate the results
randActTab = tabulate(randActiInter);
actTab = tabulate(activeInter);

figure
hold on
plot(actTab(:,1),actTab(:,3)/actTab(1,3),'LineWidth',3)
plot(randActTab(:,1),randActTab(:,3)/randActTab(1,3),'LineWidth',3)
hold off
box on
legend('Model','Random')
xlim([1,20])
xlabel('$\Delta t$','Interpreter','latex','FontSize',28)
ylabel('Fraction','FontSize',28)
set(gca,'FontSize',24,'LineWidth',1,'XScale','log')

% histogram of all the silent interval
silentIt = figure;
set(silentIt,'color','w','Units','inches')
pos(3)=3.8;  %17.4; %8.5;%pos(3)*1.5;
pos(4)=2.8;%pos(4)*1.5;
set(silentIt,'Position',pos)

histogram(silentInter(:))
xlabel('Silent interval','FontSize',16)
ylabel('Count','FontSize',16)
set(gca,'LineWidth',1,'FontSize',16)

% prefix = 'Ring_Silent_Interval';
% saveas(LtoR,[sFolder,filesep,prefix,'.fig'])
% print('-depsc',[saveFolder,filesep,prefix,'.eps'])

% histogram of the active period
activeIt = figure;
set(activeIt,'color','w','Units','inches')
pos(3)=3.8;  %17.4; %8.5;%pos(3)*1.5;
pos(4)=2.8;%pos(4)*1.5;
set(activeIt,'Position',pos)
histogram(activeInter(:))
xlim([0,100])
xlabel('Active interval','FontSize',16)
ylabel('Count','FontSize',16)
set(gca,'LineWidth',1,'FontSize',16)

% comaparison with the random permutation
actInxPerm = reshape(activeInx(randperm(size(activeInx,1)*size(activeInx,2))),size(activeInx));


% indicate if neurons are active
figure
imagesc(peakVal,[0,0.25])
xlabel('Time','FontSize',24)
ylabel('Neuron','FontSize',24)
set(gca,'LineWidth',1.5,'FontSize',24)


% ================change of representaitonal correlation ========

posiInx = [200,400,800];   % randomly select positions to look at
% Ysub = Yt(:,posiInx,:);
figure
for i = 1:length(posiInx)
    M = squeeze(Yt(:,posiInx(i),:));
    C = cov(M);
    subplot(1,length(posiInx),i)
    imagesc(C)
    colorbar
end
figure
plot(C(1,:))
xlabel('peak ampitude','FontSize',24)
ylabel('count','Interpreter','latex','FontSize',24)

%% Change of weight matrices, check if they reach stationary states
