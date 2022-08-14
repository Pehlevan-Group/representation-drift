% Prepare natural scence patches
% data is downloaded from http://www.rctn.org/bruno/sparsenet/, which was
% originally used in the paper Olshausen and Field in Nature, vol. 381, pp. 607-609.

close all
clear

%% configuration when running on cluster
% addpath('/n/home09/ssqin/supervised')
% start the parallel pool with 12 workers
% parpool('local', str2num(getenv('SLURM_CPUS_PER_TASK')));

plotFlag = 1;   % 1 for plot 0 for no-plot

%% setting for the graphics
% plot setting
% defaultGraphicsSetttings
% blues = brewermap(11,'Blues');
% rb = brewermap(11,'RdBu');
% set1 = brewermap(9,'Set1');
% 
% saveFolder = './figures';

%% prepare 16*16 patches
dFile = './data/IMAGES.mat';
load(dFile)
Ldim = 16;   % 16*16 patches
dims = size(IMAGES);
extraWhitten = 1;
% imgPatch = nan(Ldim*Ldim,floor(dims(1)/Ldim)*floor(dims(2)/Ldim)*dims(3));
% 
% count = 1;
% for i = 1:dims(3)
%     for j = 1:floor(dims(1)/Ldim)
%         for k = 1: floor(dims(2)/Ldim)
%             temp = IMAGES((j-1)*Ldim + 1:j*Ldim,(k-1)*Ldim+1:k*Ldim,i);
%             imgPatch(:,count) = temp(:);
%             count = count + 1;
%         end
%     end
% end

totPatch = 4e4;  % total number of patches sampled
imgPatch = nan(Ldim*Ldim,totPatch);

for i = 1:totPatch
    xinx = randperm(Ldim*(Ldim-1) + 1,1);
    yinx = randperm(Ldim*(Ldim-1) + 1,1);
    zinx = randperm(dims(3),1);
    temp = IMAGES(xinx: xinx + Ldim - 1,yinx:yinx+Ldim - 1,zinx);
    imgPatch(:,i) = temp(:);
end

% extra whittening
if extraWhitten
    avg = mean(imgPatch,1);
    X = imgPatch - repmat(avg, size(imgPatch, 1), 1);
    sigma = X * X' / size(X, 2);
    [U,S,V] = svd(sigma);
    epsilon = 1e-5;
    imgPatch = U * diag(1./sqrt(diag(S) + epsilon)) * U' * X;
end



% visualize the image patches
if plotFlag
    numPatchSel = 10*10;
    sel_inx = randperm(size(imgPatch,2),numPatchSel);
    figure
    hp = tight_subplot(10,10);
    % sel_inx = randperm(param.Np, min(100,param.Np));
    for i = 1:length(sel_inx)
        fm = reshape(imgPatch(:,sel_inx(i)), Ldim, Ldim);
        imagesc(hp(i),fm);
        colormap('gray')
        hp(i).XAxis.Visible = 'off';
        hp(i).YAxis.Visible = 'off';
    end
end

% save the randomply sampled img patches
% save('./data/imgPatch.mat','imgPatch')

%% setup the learning parameters

n = size(imgPatch,1);   % number of neurons
k = 256;   % 200 output neurons
t = size(imgPatch,2);   % total number of frames
BatchSize = 1;

noiseStd = 1e-3; % 0.005 for ring, 1e-3 for Tmaze
learnRate = 5e-4;
params.gy = 0.1;   % update step for y

X = imgPatch;
Cx = X*X'/t;  % input covariance matrix

% store the perturbations
record_step = 100;

% initialize the states
y0 = zeros(k,BatchSize);


% Use offline learning to find the inital solution
params.W = randn(k,n)/Ldim;
params.M = eye(k);   % lateral connection if using simple nsm
params.lbd1 = 0;     % regularization for the simple nsm, 1e-3
params.lbd2 = 0;   %default 1e-3

params.alpha = 0;  % default 0.9
params.b = zeros(k,1);  % biase
params.learnRate = learnRate;  % learning rate for W and b
params.noise =  noiseStd;   % stanard deivation of noise 

% noise1 = noiseStd;
% noise2 = 0;
for i = 1:2e5
    x = X(:,randperm(t,BatchSize));  % randomly select one input
    [states, params] = MantHelper.nsmDynBatch(x,y0, params);
    y = states.Y;
    % using dynamic learnig rate as Cengiz did
    if i <= 1e4
        params.learnRate = 1e-3;
    elseif i <= 1e5
        params.learnRate = 1e-4;
    else
        params.learnRate = 1e-5;
    end
        
    % update weight matrix
    params.W = (1-params.learnRate)*params.W + params.learnRate*y*x'/BatchSize...
        +sqrt(params.learnRate)*params.noise*randn(k,n);        
    params.M = (1-params.learnRate)*params.M + params.learnRate*y*y'/BatchSize;
%         params.M = (1-params.learnRate)*params.M + params.learnRate*y*y'/BatchSize + ...
%             sqrt(params.learnRate)*params.noise*randn(k,k);
    params.b = (1-params.learnRate)*params.b + params.learnRate*sqrt(params.alpha)*mean(y,2);
end

% check the receptive field
% Xsel = X(:,1:10:end);
Y0 = zeros(k,size(X,2));
[states_fixed_nn, params] = MantHelper.nsmDynBatch(X,Y0,params);

% save the data
% save(['./data/naturalScene',date,'.mat'],)

% =============tuning curve of neurons ==========
figure
imagesc(states_fixed_nn.Y,[0,0.5])
colorbar

% ============ learned feature map ======================
numFeature = 10*10;
sel_inx = randperm(k,numFeature);
figure
hp = tight_subplot(10,10);
% sel_inx = randperm(param.Np, min(100,param.Np));
for i = 1:length(sel_inx)
    fm = reshape(params.W(sel_inx(i),:), Ldim, Ldim);
    imagesc(hp(i),fm);
    colormap('gray')
    hp(i).XAxis.Visible = 'off';
    hp(i).YAxis.Visible = 'off';
end


% using the reverse correlation to estimate feature map
F = states_fixed_nn.Y*imgPatch'/size(imgPatch,2);
numFeature = 8*8;
sel_inx = randperm(k,numFeature);
figure
hp = tight_subplot(8,8,[.01 .01],[.01 .01],[.01 .01]);
% sel_inx = randperm(param.Np, min(100,param.Np));
for i = 1:length(sel_inx)
    fm = reshape(F(sel_inx(i),:), Ldim, Ldim);
    imagesc(hp(i),fm);
    colormap('gray')
    hp(i).XAxis.Visible = 'off';
    hp(i).YAxis.Visible = 'off';
end


%% check the change of feature map
% first, get the response based on the parameters
% only use the last 50 time points
numFeature = 8*8;
tot = length(allW);
timeSel = [100, 125, 150];
sel_inx = randperm(k,numFeature);  % randomly select 8*8 neurons
sel_x = randperm(totPatch,20);     % randomly select 20 input patches
for j = 1:length(timeSel)
    
    % choose the parameters
    params.W = allW{timeSel(j)};
    params.M = allM{timeSel(j)};
    params.b = allb{timeSel(j)};
    
    Y0 = zeros(k,size(imgPatch,2));
    [states_fixed_nn, params] = MantHelper.nsmDynBatch(imgPatch,Y0,params);
    
    F = states_fixed_nn.Y*imgPatch'/size(imgPatch,2);
    
    % feature maps
    figure
    hp = tight_subplot(8,8,[.01 .01],[.01 .01],[.01 .01]);
    for i = 1:length(sel_inx)
        fm = reshape(F(sel_inx(i),:), Ldim, Ldim);
%         fm = reshape(params.W(sel_inx(i),:), Ldim, Ldim);
        imagesc(hp(i),fm);
        colormap('gray')
        hp(i).XAxis.Visible = 'off';
        hp(i).YAxis.Visible = 'off';
    end
    
    %similarity matrix
    Y_sel = states_fixed_nn.Y(:,sel_x);
    SM = Y_sel'*Y_sel/size(Y_sel,2);
    figure
    imagesc(Y_sel)
    colorbar
    
    figure
    imagesc(SM)
    colorbar

end

%% Corrleation of feature map, response compared with time point 0
timeSep = 5;
startFrame = 100;



% =========== Example place field ========================
% sort and find the
sel_inx = neurOrder(10:round(k/7):k);
rfExamples = states_fixed_nn.Y(sel_inx,:);
figure
for i = 1:length(sel_inx)
    plot(states_fixed_nn.Y(sel_inx(i),:),'Color',set1(i,:),'LineWidth',3)
    hold on
end
hold off
xlabel('frame','FontSize',28)
ylabel('response','FontSize',28)
set(gca,'FontSize',24)

%============= similarity matrix of ordered represenation =================
SM = states_fixed_nn.Y(neurOrder,:)'*states_fixed_nn.Y(neurOrder,:);
figure
imagesc(SM,[0,0.5])
colorbar
% ax = gca;
% ax.XTick = [1 500 1000];
% ax.YTick = [1 500 1000];
% ax.XTickLabel = {'0', '\pi', '2\pi'};
% ax.YTickLabel = {'0', '\pi', '2\pi'};
xlabel('frame','FontSize',28)
ylabel('frame','FontSize',28)
set(gca,'FontSize',24)


%% continue updating with noise
%{
% BatchSize = 1;  % this used only during debug 
tot_iter = 1e3;
num_sel = 200;
step = 2;
time_points = round(tot_iter/step);


% store the weight matrices to see if they are stable
% allW = nan(k,n,time_points);
% allM = nan(k,k,time_points);

% testing data, only used when check the representations
Xsel = X(:,1:2:end);   % modified on 7/18/2020
Y0 = zeros(k,size(Xsel,2));


% Z0 = zeros(m,size(Xsel,2));
Yt = nan(k,size(Xsel,2),time_points);

for i = 1:tot_iter
    x = X(:,randperm(t,BatchSize));  % randomly select one input
    [states, params] = MantHelper.nsmDynBatch(x,y0,params);
    y = states.Y;

    % update weight matrix
    params.W = (1-params.learnRate)*params.W + params.learnRate*y*x'/BatchSize + ...
        sqrt(params.learnRate)*params.noise*randn(k,n);  % adding noise
    params.M = (1-params.learnRate)*params.M + params.learnRate*y*y'/BatchSize;
    params.b = (1-params.learnRate)*params.b + params.learnRate*sqrt(params.alpha)*mean(y,2);

    % store and check representations
    if mod(i, step) == 0
        [states_fixed, params] = MantHelper.nsmDynBatch(Xsel,Y0,params);
        Yt(:,:,round(i/step)) = states_fixed.Y;
%         allW(:,:,round(i/step)) = params.W;
    end
end 


%% Analysis, check the change of place field
pkThreshold = 0.02;  % active threshold
blues = brewermap(11,'Blues');

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

% =========place field order by the begining ======================
inxSel = [100, 250, 500];
% inxSel = [100,150,200];
figure
for i = 1:length(inxSel)
    subplot(1,3,i)
    imagesc(Yt(:,:,inxSel(i)),[0,0.4])
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
    imagesc(Yt(neuroInx,:,inxSel(i)),[0,0.5])
    colorbar
    ax = gca;
    ax.XTick = [1 500 1000];
    ax.XTickLabel = {'0', '\pi', '2\pi'};
    title(['iteration ', num2str(inxSel(i))])
    xlabel('position')
    ylabel('sorted index')
end

% ordered by time point 1
% inxSelPV = [200,300,400];
[~,neuroInx] = sort(peakInx(:,inxSel(1)));

figure
for i = 1:length(inxSel)
    subplot(1,3,i)
    imagesc(Yt(neuroInx,:,inxSel(i)),[0,0.5])
    colorbar
    ax = gca;
    ax.XTick = [1 500 1000];
    ax.XTickLabel = {'0', '\pi', '2\pi'};
    title(['iteration ', num2str(inxSel(i))])
    xlabel('position')
    ylabel('sorted index')
end

% change of tunig curve
curveSel = randperm(k,1);  % randomly slect one
plot(Yt(curveSel,:,100))
figure
hold on
for i = 1:length(inxSel)
    plot((1:2:201)',Yt(curveSel,:,inxSel(i))','Color',blues(1+3*i,:))
end
hold off
box on
legend(['t=',num2str(inxSel(1))],['t=',num2str(inxSel(2))],['t=',num2str(inxSel(3))])
xlabel('Frame','FontSize',24)
ylabel('Response','FontSize',24)
set(gca,'LineWidth',1.5,'FontSize',20)


corrTunCurve = nan(size(Yt,3),1);
for i = 1:length(corrTunCurve)
    C = corrcoef(Yt(curveSel,:,i)',Yt(curveSel,:,1)');
    corrTunCurve(i) = C(2,1);
end
figure
plot(corrTunCurve)
xlabel('time','FontSize',24)
ylabel('Corr. Coeff.','FontSize',24)
set(gca,'FontSize',20,'LineWidth',1.5)


% correlation of population vector
fmSel = randperm(size(Yt,2),1); % randomly slect three frames to consider
pvCorr = zeros(size(Yt,3),size(Yt,2)); 
[~,neuroInx] = sort(peakInx(:,inxSel(1)));

for i = 1:size(Yt,3)
    for j = 1:size(Yt,2)
        temp = Yt(:,j,i);
        C = corrcoef(temp,Yt(:,j,1));
        pvCorr(i,j) = C(1,2);
    end
end

figure
fh = shadedErrorBar((1:size(pvCorr,1))',pvCorr',{@mean,@std});
box on
set(fh.edge,'Visible','off')
fh.mainLine.LineWidth = 4;
fh.mainLine.Color = blues(10,:);
fh.patch.FaceColor = blues(7,:);
ylim([0.25,1])
xlabel('Time','FontSize',24)
ylabel('PV correlation','FontSize',24)
set(gca,'FontSize',20,'LineWidth',1.5)


% PV similarity
pvSM = nan(size(Yt,3),2);  % store the average and std
for i= 1:size(Yt,3)
    temp = sum(Yt(:,:,i).*Yt(:,:,1),1);
    pvSM(i,:) = [mean(temp),std(temp)];
end

figure
fh = shadedErrorBar((1:size(pvSM,1))',pvSM(:,1),pvSM(:,2));
box on
set(fh.edge,'Visible','off')
fh.mainLine.LineWidth = 4;
fh.mainLine.Color = blues(10,:);
fh.patch.FaceColor = blues(7,:);
ylim([0.1,0.4])
xlabel('Time','FontSize',24)
ylabel('PV similarity','FontSize',24)
set(gca,'FontSize',20,'LineWidth',1.5)



% merged left and right
% YtMerge = cat(1,Yt(neuroInx(1:round(k/2)),1:500,:),Yt(neuroInx((k/2+1):end),501:1000,:));
% mergePeakInx = nan(k,time_points);
% for i = 1:time_points
%     [~,mergePeakPosi] = sort(YtMerge(:,:,i),2,'descend');
%     mergePeakInx(:,i) = mergePeakPosi(:,1);
% end
% [~, newInx] = sort(mergePeakInx(:,inxSelPV(1)));
% 
% figure
% for i = 1:length(inxSelPV)
%     subplot(1,3,i)
%     imagesc(YtMerge(newInx(100:k),:,inxSelPV(i)),[0,0.4])
%     colorbar
%     ax = gca;
%     ax.XTick = [1 500 1000];
%     ax.XTickLabel = {'0', '\pi', '2\pi'};
%     title(['iteration ', num2str(inxSelPV(i))])
%     xlabel('position')
%     ylabel('sorted index')
% end


% ======== place field of a single neurons across repeats =============
% figure
% epLoc = [1, 100, 150];
% for i = 1:3
%     ypart = squeeze(Yt(epLoc(i),1:500,:));
%     subplot(1,3,i)
%     imagesc(ypart')
%     colorbar
%     xlabel('location')
%     ylabel('time')
% %     set(gca,'XTick',100:100:500,'XTickLabel',['0.1','0.2','0.3','0.4','0.5'])
%     set(gca,'XTick',[0, 250, 500],'XTickLabel',{'0','0.5','1'})
%     title(['neuron ',num2str(epLoc(i))])
% end


% ======== representation similarity matrix =======
figure
for i = 1:length(inxSel)
%     [~,neuroInx] = sort(peakInx(:,inxSel(i)));
    SM = Yt(:,:,inxSel(i))'*Yt(:,:,inxSel(i));
    subplot(1,3,i)
    imagesc(SM)
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

% check the shift
figure
plot(newPeaks(1:5,:)')
xlabel('iteration','FontSize',28)
ylabel('shift of RF','FontSize',28)
ax =  gca;
ax.YTick = (-3:1:3)*pi;
ax.YTickLabel = {'-3\pi','-2\pi', '-\pi','0', '\pi', '2\pi','3\pi'};
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
quantiles = (1:3)/4*pi;
probs = nan(length(tps),length(quantiles));
for i = 1:length(tps)
    diffLag = min(pi, abs(pks0(:,tps(i)+1:end,:) - pks0(:,1:end-tps(i),:)));
    for j = 1:length(quantiles)
        probs(i,j) = sum(diffLag(:)>quantiles(j))/length(diffLag(:));
    end
end
figure
plot(tps'*step,probs,'o-','MarkerSize',10)
legend('>1/4L','>1/2L','>3/4L')
xlabel('$\Delta t$','Interpreter','latex','FontSize',24)
ylabel('Probability','FontSize',24)
set(gca,'LineWidth',1.5,'FontSize',24)




% estimate the diffusion constant
msds = nan(floor(time_points/2),k);
for i = 1:floor(time_points/2)
    diffLag = newPeaks(:,i+1:end,:) - newPeaks(:,1:end-i,:);
    msds(i,:) = mean(diffLag.*diffLag,2);
end


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

%% Change of weight matrices, check if they reach stationary state
%}