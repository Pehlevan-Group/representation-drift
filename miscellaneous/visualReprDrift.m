% representational drift in visual systems
% nonnegtive similarity mathching and drifting representation of natural
% movie, related to Deitch, Rubin and Ziv bioRxiv 2020 paper "Representational 
% drift in the mouse visual cortex" 
% the input data is from a movie " touch of the evil", we only slect some
% of the first 30s frames

close all
clear

%% Prepare the data
%{
% load and get the frame of video
videoName = './data/TouchofEvil.mp4';
obj = VideoReader(videoName);
video = obj.read();
vSize = size(video);
scaleFactor = 0.15;  % down scale the image size
% selIndices = 600:15:7000;
selIndices = 3450:2:(3450+900);
newData = zeros(round(vSize(1)*scaleFactor),round(vSize(2)*scaleFactor),...
    length(selIndices));

for i = 1:length(selIndices)
    temp = rgb2gray(video(:,:,:,selIndices(i)));
    % resize
    fmscale = imresize(temp,scaleFactor);
    
    % normalize to unit norm 
    newData(:,:,i) = double(fmscale)/norm(double(fmscale));
end
% save the data
% sFile = './data/naturalMovie.mat';
sFile = './data/naturalMovie_exp.mat';
save(sFile,'newData')
%}

%% Use the same data as in the experiment
% load and get the frame of video
%{
videoName = './data/TouchofEvil_Exp.mat';
load(videoName)
% obj = VideoReader(videoName);
% video = obj.read();
vSize = size(TouchofEvil);
scaleFactor = 0.5;  % down scale the image size
selIndices = 3465:2:(3465+900);
newData = zeros(round(vSize(1)*scaleFactor),round(vSize(2)*scaleFactor),...
    length(selIndices));

for i = 1:length(selIndices)
%     temp = rgb2gray(video(:,:,:,selIndices(i)));
    temp = TouchofEvil(:,:,selIndices(i));
    % resize
    fmscale = imresize(temp,scaleFactor);
    
    % normalize to unit norm 
    newData(:,:,i) = double(fmscale)/norm(double(fmscale));
end
% save the data
sFile = './data/naturalMovie_exp.mat';
save(sFile,'newData')

%}


%% setting for the graphics

% plot setting
% defaultGraphicsSetttings
blues = brewermap(13,'Blues');
reds = brewermap(13,'Reds');
greens = brewermap(13,'Greens');
greys = brewermap(11,'Greys');

rb = brewermap(11,'RdBu');
set1 = brewermap(9,'Set1');

spectralMap = flip(brewermap(256,'Spectral'),1);


saveFolder = './figures';

%% noisy non-negative similarity matching

% load the data
% load('./data/naturalMovie_exp.mat');
load('./data/gratings.mat');          % generated gratings
newData = img;
% load('./data/naturalMovieLong.mat');

X = reshape(newData,size(newData,1)*size(newData,2),size(newData,3));
SMX = X'*X;

% visualize the data by dimensionality reduction
% X = Yt(:,:,1);
Xred = tsne(X');
spectrColor = brewermap(size(Xred,1),'Spectral');

figure
hold on
for i = 1:size(Xred,1)
    plot(Xred(i,1),Xred(i,2),'o','Color',spectrColor(i,:),'MarkerSize',8)
end
hold off
box on
xlabel('tsne1')
ylabel('tsne2')

% multidimensional scaling
% load the data from python sklearn
dFile = './data/natural_movie_mds_long.csv';
Xmds = csvread(dFile,1,0);
% Dm = squareform(pdist(X','euclidean')); % distance matrix
% Xmds = cmdscale(Dm,2);
spectrColor = brewermap(size(Xmds,1),'Spectral');

figure
hold on
for i = 1:size(Xmds,1)
    plot(Xmds(i,1),Xmds(i,2),'o','Color',spectrColor(i,:),'MarkerSize',8,'LineWidth',1.5)
end
hold off
box on
xlabel('mds1')
ylabel('mds2')
colormap(spectralMap)
colorbar




% tSNE
% dotColors = flip(brewermap(size(Xred,1),'Spectral'));
% figure
% hold on
% for i = 1:size(Xred,1)
%     plot(Xred(i,1),Xred(i,2),'o','Color',dotColors(i,:))
% end
% hold off
% box on
% xlabel('t-SNE1','FontSize',24)
% ylabel('t-SNE2','FontSize',24)
% set(gca,'FontSize',20,'LineWidth',1.5)

% PCA
% [COEFF, SCORE, ~, ~,EXPLAIN] = pca(X');
% spectrColor = brewermap(size(X,2),'Spectral');
% figure
% hold on
% for i = 1:size(Xred,1)
%     plot(SCORE(i,1),SCORE(i,2),'o','Color',spectrColor(i,:),'MarkerSize',8)
% end
% hold off
% box on
% xlabel(['PC1 (',num2str(round(EXPLAIN(1)*100)/100),'%)'])
% ylabel(['PC1 (',num2str(round(EXPLAIN(2)*100)/100),'%)'])
% 
% 
% % LLE
% Xlle = lle(X,10,2);
% figure
% hold on
% for i = 1:size(Xlle,2)
%     plot(Xlle(1,i),Xlle(2,i),'o','Color',dotColors(i,:))
% end
% hold off
% box on
% xlabel('LLE-1','FontSize',24)
% ylabel('LLE-2','FontSize',24)
% set(gca,'FontSize',20,'LineWidth',1.5)




figure
% imagesc(SMX,[0.3,1.2]);
imagesc(SMX,[100,1000]);
colorbar
ax.XTick = [1 5e3 1e4];
ax.YTick = [1 5e3 1e4];
ax.XTickLabel = {'0', '\pi', '\pi'};
ax.YTickLabel = {'0', '\pi', '\pi'};
xlabel('position','FontSize',28)
ylabel('position','FontSize',28)
set(gca,'FontSize',24)
title('$X^{\top}X$','Interpreter','latex')

%% setup the learning parameters

n = size(newData,1)*size(newData,2);   % number of neurons
dimOut = 200;            % 200 output neurons
t = size(newData,3);   % total number of frames
BatchSize = 1;

noiseStd = 1e-5;       % 1e-5
learnRate = 0.02;      % learning rate 0.05
params.gy = 0.05;        % update step for y



Cx = X*X'/t;  % input covariance matrix

% store the perturbations
record_step = 100;

% initialize the states
y0 = zeros(dimOut,BatchSize);


% Use offline learning to find the inital solution
params.W = 0.01*randn(dimOut,n);
params.M = eye(dimOut);      % lateral connection if using simple nsm
params.lbd1 = 1e-3;         % regularization for the simple nsm, 1e-3
params.lbd2 = 1e-3;         %default 1e-3

params.alpha = 0;        % default 0.92
params.b = zeros(dimOut,1);  % bias
params.learnRate = learnRate;  % learning rate for W and b
params.noise =  noiseStd;   % stanard deivation of noise 


% initial phase, find the solution
for i = 1:2e3
    % sequential input
    inx = mod(i,size(X,2));
    if inx ==0
        inx = size(X,2);
    end
%     x = X(:,randperm(t,BatchSize));  % randomly select one input
    x = X(:,inx);
    [states, params] = MantHelper.nsmDynBatch(x,y0, params);
    y = states.Y;

    % update weight matrix, only W with noise
    params.W = (1-params.learnRate)*params.W + params.learnRate*y*x'/BatchSize...
        +sqrt(params.learnRate)*params.noise*randn(dimOut,n);        
    params.M = (1-params.learnRate)*params.M + params.learnRate*y*y'/BatchSize;
%         params.M = (1-params.learnRate)*params.M + params.learnRate*y*y'/BatchSize + ...
%             sqrt(params.learnRate)*params.noise*randn(k,k);
    params.b = (1-params.learnRate)*params.b + params.learnRate*sqrt(params.alpha)*mean(y,2);
end

% check the receptive field
Y0 = zeros(dimOut,size(X,2));
[states_fixed_nn, params] = MantHelper.nsmDynBatch(X,Y0,params);


% ============= Y: output gram heatmap, before ordering ==========
figure
imagesc(states_fixed_nn.Y,[0,40])
% imagesc(states_fixed_nn.Y,[0,0.5])
colorbar
xlabel('frame','FontSize',28)
ylabel('neuron','FontSize',28)
set(gca,'FontSize',24)

% ============= Y: output gram heatmap after ordering ==========
% sort the location
[sortVal,sortedInx] = sort(states_fixed_nn.Y,2,'descend');
[~,neurOrder] = sort(sortedInx(:,1));
figure
imagesc(states_fixed_nn.Y(neurOrder,:),[0,0.5])
colorbar
xlabel('position','FontSize',28)
ylabel('neuron','FontSize',28)
ax = gca;
set(gca,'FontSize',24)

% =========== peak amplitude distribution ================

figure
histogram(sortVal(:,1),30)

% =========== Example tuning curves ========================
% sort and find the
sel_inx = neurOrder(10:round(dimOut/5):dimOut);
rfExamples = states_fixed_nn.Y(sel_inx,:);

figure
for i = 1:length(sel_inx)
    plot(states_fixed_nn.Y(sel_inx(i),:),'Color',set1(i,:),'LineWidth',1.5)
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
xlabel('frame','FontSize',28)
ylabel('frame','FontSize',28)
set(gca,'FontSize',24)


%% continue updating with noise

tot_iter = 2e3;
num_sel = 200;
step = 4;
time_points = round(tot_iter/step);

% testing data, only used when check the representations
Xsel = X;        % modified on 10/21/2020
Y0 = zeros(dimOut,size(Xsel,2));


% store all the output gram at each time point
Yt = nan(dimOut,size(Xsel,2),time_points);

for i = 1:tot_iter
    inx = mod(i,size(X,2));
    if inx ==0
        inx = size(X,2);
    end
    x = X(:,inx);
%     x = X(:,randperm(t,BatchSize));  % randomly select one input
    [states, params] = MantHelper.nsmDynBatch(x,y0,params);
    y = states.Y;

    % update weight matrix
    params.W = (1-params.learnRate)*params.W + params.learnRate*y*x'/BatchSize + ...
        sqrt(params.learnRate)*params.noise*randn(dimOut,n);  % adding noise
    params.M = (1-params.learnRate)*params.M + params.learnRate*y*y'/BatchSize;
    params.b = (1-params.learnRate)*params.b + params.learnRate*sqrt(params.alpha)*mean(y,2);

    % store and check representations
    if mod(i, step) == 0
        [states_fixed, params] = MantHelper.nsmDynBatch(Xsel,Y0,params);
        Yt(:,:,round(i/step)) = states_fixed.Y;
    end
end 


%% Analysis, check the change of place field
pkThreshold = 0.02;     % active threshold

% peak of tuninig curve
peakInx = nan(dimOut,time_points);
peakVal = nan(dimOut,time_points);
for i = 1:time_points
    [pkVal, peakPosi] = sort(Yt(:,:,i),2,'descend');
    peakInx(:,i) = peakPosi(:,1);
    peakVal(:,i) = pkVal(:,1);
end

% ======== faction of neurons have receptive field at a give time =====
% quantified by the peak value larger than a threshold 0.01
rfIndex = peakVal > pkThreshold;

% fraction of neurons
activeRatio = sum(rfIndex,1)/dimOut;
figure
plot(101:time_points,activeRatio(101:end))
ylim([0,1])
xlabel('iteration')
ylabel('active fraction')

% histogram of the fraction of activation
figure
histogram(activeRatio(101:end))

% drop in 
figure
plot(sum(rfIndex,2)/dimOut)

% =========place field order by the begining ======================

inxSel = [100, 250, 500];
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

% =========== ordered by time point 1 ========
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

% ************************************************
% change of tunig curve
% ************************************************

curveSel = randperm(dimOut,3);   % randomly slect three neurons

% plot the tuning curve at three time points
figure
hold on
for i = 1:length(inxSel)
    plot((1:size(Yt,2))',Yt(curveSel(3),:,inxSel(i))','Color',greens(2+3*i,:))
end
hold off
box on
legend(['t=',num2str(inxSel(1))],['t=',num2str(inxSel(2))],['t=',num2str(inxSel(3))])
xlabel('Frame','FontSize',24)
ylabel('Response','FontSize',24)
set(gca,'LineWidth',1.5,'FontSize',20)

% *********************************************
% tuning property overlay on 2D manifold, t-SNE
% *********************************************
% define the colors, which indicate strength
colorArray = nan(10,3,length(curveSel));
colorArray(:,:,1) = blues(3:12,:);
colorArray(:,:,2) = reds(3:12,:);
colorArray(:,:,3) = greens(3:12,:);


Xtsne = tsne(Xsel');
for i = 1:length(inxSel)
    figure
    hold on
    % tnse
    plot(Xtsne(:,1),Xtsne(:,2),'o','Color',greys(5,:),'MarkerSize',8,'LineWidth',1.5)
    
    for ne = 1:length(curveSel)
        posi = find(Yt(curveSel(ne),:,inxSel(i)) > 0.01); % active neurons
        vals = Yt(curveSel(ne),posi,inxSel(i));

        % mapped to colors
        colors = ceil(10*(vals./max(vals)));
        for j=1:length(vals)
            plot(Xtsne(posi(j),1),Xtsne(posi(j),2),'o','MarkerSize',8,'MarkerFaceColor',...
                colorArray(colors(j),:,ne),'MarkerEdgeColor',colorArray(colors(j),:,ne))
        end
    end
    hold off
    box on
    title(['t=', num2str(inxSel(i))])
    xlabel('t-SNE1')
    ylabel('t-SNE2')

end


% ******************************
% projection on to the MDS
% ******************************
curveSel = randperm(dimOut,3);   % randomly slect three neurons
Xemb = Xmds;
for i = 1:length(inxSel)
    figure
    hold on
    % tnse
    plot(Xemb(:,1),Xemb(:,2),'o','Color',greys(5,:),'MarkerSize',8,'LineWidth',1.5)
    
    for ne = 1:length(curveSel)
        posi = find(Yt(curveSel(ne),:,inxSel(i)) > 0.01); % active neurons
        vals = Yt(curveSel(ne),posi,inxSel(i));

        % mapped to colors
        colors = ceil(10*(vals./max(vals)));
        for j=1:length(vals)
            plot(Xemb(posi(j),1),Xemb(posi(j),2),'o','MarkerSize',8,'MarkerFaceColor',...
                colorArray(colors(j),:,ne),'MarkerEdgeColor',colorArray(colors(j),:,ne))
        end
    end
    hold off
    box on
    title(['t=', num2str(inxSel(i))])
    xlabel('mds1')
    ylabel('mds2')
end

% *************************************
% Change of center of mass, example Neuron
% *************************************
neuronSel = curveSel(3);  % index of example neuron
centerMass = nan(size(Yt,3),3, dimOut); % x,y and weight, all the neurons
for i = 1:size(Yt,3)
    for j=1:dimOut
        posi = find(Yt(j,:,i) > 0.05); % active neurons
        if ~isempty(posi)
            vals = Yt(j,posi,i);
            preFactor = abs(vals')./sum(vals)*ones(1,2);
            xyCoord = Xmds(posi,:);
            centerMass(i,:,j) = [sum(preFactor.*xyCoord,1),mean(abs(vals))];
        end
    end
end

% visulization
% neuronSel = 120;
spectrColor = brewermap(size(centerMass,1),'Spectral');

figure
hold on
plot(Xmds(:,1),Xmds(:,2),'o','Color',greys(5,:),'MarkerSize',8,'LineWidth',1.5)

for i = 1:size(centerMass,1)
    if ~isnan(centerMass(i,1,neuronSel))
        plot(centerMass(i,1,neuronSel),centerMass(i,2,neuronSel),'o','MarkerSize',8,'MarkerFaceColor',...
                spectrColor(i,:),'MarkerEdgeColor',spectrColor(i,:))
    end
end
%
box on
hold off
xlabel('mds1')
ylabel('mds2')

%%
% ***********************************************************
% make a video of the place field, projected on 2D manifold
% ***********************************************************
figure
set(gcf,'color','w');
hold on
plot(Xmds(:,1),Xmds(:,2),'o','Color',greys(4,:),'MarkerSize',8,'LineWidth',1.5)
for i = 1:size(blues,1)
    plot(Xmds(i,1),Xmds(i,2),'o','Color',blues(i,:),'MarkerSize',8,'LineWidth',1.5)
end
box on
xlabel('mds1')
ylabel('mds2')

detlaT = 0.05;
f = getframe(gcf);
[im,map] = rgb2ind(f.cdata,256,'nodither');
k = 1;
for i = 1:size(Yt,3)
    plot(Xmds(:,1),Xmds(:,2),'o','Color',greys(4,:),'MarkerSize',8,'LineWidth',1.5)
    hold on
    posi = find(Yt(neuronSel,:,i) > 0.01); % active neurons
    if ~isempty(posi)
        vals = Yt(neuronSel,posi,i);
        % mapped to colors
        colors = ceil(10*(vals./max(vals)));
        
        for j=1:length(vals)
            plot(Xmds(posi(j),1),Xmds(posi(j),2),'o','MarkerSize',8,'MarkerFaceColor',...
                blues(2+colors(j),:),'MarkerEdgeColor',blues(2+colors(j),:))
        end
        
    end
        
    hold off
    box on
    xlim([-1,1])
    ylim([-1,1])
    xlabel('mds1')
    ylabel('mds1')
%     set(gca,'LineWidth',1.5,'FontSize',20)
    title(['time = ', num2str(i*step)])
    f = getframe(gcf);
    im(:,:,1,k) = rgb2ind(f.cdata,map,'nodither');
    k = k + 1;  
end
imwrite(im,map,'naturaMovieRF_manifold2.gif','DelayTime',detlaT,'LoopCount',inf)

%% stability of tuninig curve, amplitude
% RF center of mass amplitude and the shift of centers

shiftStat = nan(dimOut,2);  %store the mean and std of peak shift
allShiftDist = [];
largestJump = nan(dimOut,1);  % only use the lagest three jump
for i = 1:dimOut
    flag = find(~isnan(centerMass(:,1,i)));
    
    dxs = centerMass(flag(2:end),1,i) - centerMass(flag(1:end-1),1,i);
    dys = centerMass(flag(2:end),2,i) - centerMass(flag(1:end-1),2,i);
    shiftEcul = sqrt(dxs.^2 + dys.^2);
    shiftStat(i,:) = [mean(shiftEcul),std(shiftEcul)]; 
    allShiftDist = [allShiftDist;shiftEcul];
    [~,temp] =  sort(shiftEcul,'descend');
    largestJump(i) = mean(shiftEcul(temp(1:3)));
end


% largest jump and the average response amplitude
figure
aveAmp = nanmean(squeeze(centerMass(:,3,:)),1);
plot(aveAmp,largestJump,'o')

% amplitude trajectory
figure
plot(centerMass(:,3,10))
ylabel('Ave. Resp. Ampl')
xlabel('Time')
xlim([0,500])

% distribution of all jump for all the neurons
figure
histogram(allShiftDist)
[fp,xp] = ecdf(allShiftDist);
figure
plot(log10(xp),log10(1-fp))
xlim([-2,0.5])
xlabel('$\log_{10}(\Delta d)$','Interpreter','latex')
ylabel('$\log_{10}(1-F(\Delta d))$','Interpreter','latex')

%% Stability of individual neurons during simulation
% this is a more direct comparison with experiments

% the data is download from simulation on the cluster, need to be loaded
% first

% heatmap of two snap shot
figure
imagesc(Yt(:,:,1000))
colorbar
xlabel('Time')
ylabel('Neuron')
title('t = 1000')

exampNeuron = 89;  % randomly select one neuron
tuningMat = squeeze(Yt(exampNeuron,:,:));
figure
imagesc(tuningMat')
colorbar
xlabel('Frame')
ylabel('Time')
title(['neuron ',num2str(exampNeuron)])

figure
imagesc(tuningMat(50:100,:)',[0,30])
colorbar

figure
plot(tuningMat(161,:))
xlabel('Time')
ylabel('Response')
title('frame 161')
%%

% ******************************
% correlation of tuning curve
% ******************************
corrTunCurve = nan(size(Yt,3),size(Yt,1));
for i = 1:size(Yt,3)
    for j = 1:size(Yt,1)
        C = corrcoef(Yt(j,:,i)',Yt(j,:,1)');
        corrTunCurve(i,j) = C(2,1);
    end
end

figure
plSample = randperm(size(corrTunCurve,2),20);  % randomly select 10 lines to plot
hold on
plot(corrTunCurve(:,plSample),'LineWidth',1,'Color',greys(6,:))
plot(nanmean(corrTunCurve,2),'LineWidth',4,'Color',blues(11,:))
hold off
box on
xlabel('time','FontSize',24)
ylabel('Corr. Coeff.','FontSize',24)
set(gca,'FontSize',20,'LineWidth',1.5)

% ************************************
% cumulative correlation distribution
% ************************************
figure
hold on
for i = 1:length(inxSel)
    [F,ds_x] = ecdf(corrTunCurve(inxSel(i),:));
    plot(ds_x,F,'Color',blues(3*(i+1),:))
end
hold off
box on
legend('t= 100','t=250','t=500')
xlabel('Corr. Coeff.','FontSize',24)
ylabel('ECDF','FontSize',24)
set(gca,'FontSize',20,'LineWidth',1.5)

% histogram tuning curve correlations
figure
histogram(corrTunCurve(end,:),15)
xlabel('Corr. Coeff.','FontSize',24)
ylabel('Count','FontSize',24)
set(gca,'FontSize',20,'LineWidth',1.5)
title('t = 500')

% slect three most stable neurons
[~,Istable] = sort(corrTunCurve(end,:),'descend');


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


% ======== representation similarity matrix =======
figure
for i = 1:length(inxSel)
%     [~,neuroInx] = sort(peakInx(:,inxSel(i)));
    SM = Yt(:,:,inxSel(i))'*Yt(:,:,inxSel(i));
    subplot(1,3,i)
    imagesc(SM,[0,0.5])
    colorbar
    ax = gca;
%     ax.XTick = [1 500 1000];
%     ax.YTick = [1 500 1000];
%     ax.XTickLabel = {'0', '\pi', '2\pi'};
%     ax.YTickLabel = {'0', '\pi', '2\pi'};
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



% ========== Quantify the drifting behavior ==========================
% due to the periodic condition manifold, we need to correct the movement
% using the initial order as reference point
orderedPeaks = peakInx(neurOrder,:);
shift = diff(orderedPeaks')';
reCode = zeros(size(shift));
reCode(shift > 550) = -1;
reCode(shift < -550) = 1;
addVals = cumsum(reCode,2)*2*pi;
newPeaks = orderedPeaks/1001*2*pi + [zeros(dimOut,1),addVals];

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
msds0= nan(floor(time_points/2),dimOut);
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
msds = nan(floor(time_points/2),dimOut);
for i = 1:floor(time_points/2)
    diffLag = newPeaks(:,i+1:end,:) - newPeaks(:,1:end-i,:);
    msds(i,:) = mean(diffLag.*diffLag,2);
end


diffEucl = nan(dimOut,2);  % store the diffusion constants and the exponents
for i = 1:dimOut
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

ampls = nan(dimOut,length(inxSel));
figure
for i = 1:length(inxSel)
    pinx = peakInx(:,inxSel(i));
    for j = 1:dimOut
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
