% using non-negative similarity matching to learn a 1-d place cells based
% on predefined grid cells patterns. Based on the paper lian et al 2020
% Model a linear track environemnt
% add input noise simulation
% last revised  5/11/2021

clear
close all

%% Graphics setting
% define some useful colors
blues = brewermap(11,'Blues');
reds = brewermap(11,'Reds');
BuGns = brewermap(11,'BuGn');
greens = brewermap(11,'Greens');
greys = brewermap(11,'Greys');
PuRd = brewermap(11,'PuRd');
YlOrRd = brewermap(11,'YlOrRd');
oranges = brewermap(11,'Oranges');


figWidth = 3.5;   % in unit of inches
figHeight = 2.8;
plotLineWd = 2;    %line width of plot
axisLineWd = 1;    % axis line width
labelFont = 20;    % font size of labels
axisFont = 16;     % axis font size

%% model parameters
param.ps =  200;      % number of positions along each dimension
param.Nlbd = 5;       % number of different scales/spacing,default 5
param.Nthe =  20;     % offset along the 1D
param.Ng = param.Nlbd*param.Nthe;   % total number of grid cells
param.Np = 100;       % number of place cells, default 20*20

param.baseLbd = 1/4;  % spacing of smallest grid RF, default 1/4
param.sf =  1.42;     % scaling factor between adjacent module

% parameters for learning 
noiseStd =0.01;
learnRate = 0.05;        % default 0.05

param.W = 0.5*randn(param.Np,param.Ng);   % initialize the forward matrix
param.M = eye(param.Np);        % lateral connection if using simple nsm
param.lbd1 = 0.001;              % 0.15 for 400 place cells and 5 modes
param.lbd2 = 0.05;              % 0.05 for 400 place cells and 5 modes


param.alpha = 15;               % threshold in the NSM, depends on the number of grid cells
param.gy = 0.05;                % step size of neural dynamics 
param.b = zeros(param.Np,1);    % bias vector
param.learnRate = learnRate;    % learning rate for W and b
param.noise =  noiseStd;        % stanard deivation of noise 
param.rwSpeed = 10;             %  random walk step size, only used for rand walk sampling

BatchSize = 1;                  % minibatch used to to learn
learnType = 'snsm';             % snsm, batch, online, randwalk, direction, inputNoise

makeAnimation = 0;    % whether make a animation or not

%% generate grid fields
% sample parameters for grid cells
param.lbds = param.baseLbd*(param.sf.^(0:param.Nlbd-1));   % all the spacings of different modules
param.thetas =(0:param.Nthe-1)*2*pi/param.Nthe;             % random sample rotations
% generate a Gramian of the grid fields
gridFields = nan(param.ps,param.Ng);
count = 1;    % concantenate the grid cells
for i = 1:param.Nlbd
    for j = 1: param.Nthe
        gridFields(:,count) = PlaceCellhelper.gridModule1D(param.lbds(i),...
            param.thetas(j),param.ps);
        count = count +1;
    end
end


figure
imagesc(gridFields)
xlabel('Grid cell','FontSize',24)
ylabel('position index','FontSize',24)
set(gca,'FontSize',20,'LineWidth',1.5)


%% using non-negative similarity matching to learng place fields
% generate input from grid filds

tot_iter = 2e3;         % total interation, default 2e3
sep = 20;

posiGram = eye(param.ps);
gdInput = gridFields'*posiGram;     % gramin of input matrix


% simple non-negative similarity matching
if strcmp(learnType, 'snsm')
    ystart = zeros(param.Np,BatchSize);  % inital of the output
    
    % generate position code by the grid cells
%     posiInfo = nan(tot_iter,1);
    for i = 1:tot_iter
        positions = gdInput(:,randperm(param.ps,BatchSize));

        states= PlaceCellhelper.nsmDynBatch(positions,ystart, param);
        y = states.Y;
       
        % update weight matrix, without noise
        param.W = (1-param.learnRate)*param.W + param.learnRate*y*positions'/BatchSize;
        param.M = (1-param.learnRate)*param.M + param.learnRate*y*y'/BatchSize;
%         param.W = (1-param.learnRate)*param.W + param.learnRate*y*positions'/BatchSize + ...
%             sqrt(param.learnRate)*param.noise*randn(param.Np,param.Ng);
%         param.M = max(0,(1-param.learnRate)*param.M + param.learnRate*y*y'/BatchSize + ...
%             sqrt(param.learnRate)*param.noise*randn(param.Np,param.Np));
%         param.W = max((1-param.learnRate)*param.W + param.learnRate*y*gdInput'/BatchSize,0);
        param.b = (1-param.learnRate)*param.b + param.learnRate*sqrt(param.alpha)*mean(y,2);

    end
elseif strcmp(learnType, 'randwalk')
    % position information is delivered as a random walk
    ystart = zeros(param.Np,1);  % inital of the output
    
    % generate position code by the grid cells
    posiInfo = nan(tot_iter,1);
    ix = randperm(param.ps,1);   % random select seed
    for i = 1:tot_iter
        ix = PlaceCellhelper.nextPosi1D(ix,param);
        posiInfo(i) = ix;
        positions = gdInput(:,ix);  % column-wise storation
%         positions = gdInput(:,randperm(param.ps*param.ps,1));
        
        states= PlaceCellhelper.nsmDynBatch(positions,ystart, param);
        y = states.Y;
        
        % update weight matrix
        param.M = (1-param.learnRate)*param.M + param.learnRate*y*y';
        param.W = (1-param.learnRate)*param.W + param.learnRate*y*positions' + ...
            sqrt(param.learnRate)*param.noise*randn(param.Np,param.Ng);
        param.b = (1-param.learnRate)*param.b + param.learnRate*sqrt(param.alpha)*mean(y,2);

    end
elseif strcmp(learnType, 'inputNoise')
    % position information is delivered as a random walk
    ystart = 0.1*rand(param.Np,1);  % inital of the output
    
    % generate position code by the grid cells
    posiInfo = nan(tot_iter,1);
    ix = randperm(param.ps,1);   % random select seed
    for i = 1:tot_iter
        ix = max(1,mod(i,param.ps));
        posiInfo(i) = ix;
        positions = gdInput(:,ix) + param.noise*randn(param.Ng,1);  % column-wise storage, add Gaussian noise
        
        states= PlaceCellhelper.nsmDynBatch(positions,ystart, param);
        y = states.Y;
        
        % update weight matrix
        param.M = (1-param.learnRate)*param.M + param.learnRate*y*y';
        param.W = (1-param.learnRate)*param.W + param.learnRate*y*positions';
        param.b = (1-param.learnRate)*param.b + param.learnRate*sqrt(param.alpha)*mean(y,2);

    end    
end


% estimate the place field of place cells
numProb = param.ps;    % number of positions used to estimate the RF


ys_prob = zeros(param.Np,numProb);
states= PlaceCellhelper.nsmDynBatch(gdInput,ys_prob, param);
ys = states.Y;

%% Analysis and plot

% estimate the peak positions of the place field
[~, pkInx] = sort(ys,2,'descend');

pkMat = zeros(1,param.ps);
pkMat(pkInx(:,1)) = 1;

figure
imagesc(pkMat)

[~,nnx] = sort(pkInx(:,1),'ascend');
figure
imagesc(ys(nnx,:),[0,1.5])

% amplitude of place fields
z = max(ys,[],2);
figure
histogram(z(z>0))
% xlim([2,6])
xlabel('Amplitude','FontSize',24)
ylabel('count','FontSize',24)
set(gca,'LineWidth',1.5, 'FontSize',24)


% ================= visualize the input ===========================

% input similarity
figure
imagesc(gdInput'*gdInput)
colorbar
xlabel('position index','FontSize',24)
ylabel('position index','FontSize',24)
set(gca,'LineWidth',1.5, 'FontSize',20)
% title('Input similarity matrix','FontSize',24)


% output matrix and similarity matrix
figure
imagesc(ys)
colorbar
xlabel('position index ','FontSize',24)
ylabel('place cell index','FontSize',24)
set(gca,'LineWidth',1, 'FontSize',20)
title('Output','FontSize',24)


figure
imagesc(ys'*ys)
colorbar
xlabel('position index ','FontSize',24)
ylabel('position index ','FontSize',24)
set(gca,'LineWidth',1, 'FontSize',20)
title('Output Similarity','FontSize',24)



% ============== visualize the learned matrix =======================
% feedforward connection
figure
imagesc(param.W)
colorbar
xlabel('grid cell','FontSize',24)
ylabel('place cell','FontSize',24)
set(gca,'LineWidth',1.5,'FontSize',24)

% histogram of W
figure
histogram(param.W(param.W<0.5))
xlabel('$W_{ij}$','Interpreter','latex','FontSize',24)
ylabel('Count','FontSize',24)
set(gca,'LineWidth',1.5,'FontSize',24)

% in term of individual palce cells
figure
plot(param.W(randperm(param.Np,3),:)')


% heatmap of feedforward matrix
figure
imagesc(param.W)
colorbar
xlabel('grid cell index','FontSize',24)
xlabel('place cell index','FontSize',24)
set(gca,'LineWidth',1.5, 'FontSize',20)
title('forward connection matrix','FontSize',24)


% ============== lateral connection matrix ===================
Mhat = param.M - diag(diag(param.M));
figure
imagesc(Mhat)
colorbar
xlabel('place cell index','FontSize',24)
xlabel('place cell index','FontSize',24)
set(gca,'LineWidth',1.5, 'FontSize',20)
title('Lateral connection matrix','FontSize',24)

figure
histogram(Mhat(Mhat>0.005))
xlabel('$M_{ij}$','Interpreter','latex','FontSize',24)
ylabel('Count','FontSize',24)
set(gca,'LineWidth',1.5,'FontSize',24)

%% Continuous Nosiy update
% 
tot_iter = 2e4;
num_sel = 100;
step = 10;          % subsampling
time_points = round(tot_iter/step);

pks = nan(param.Np,time_points);            % store all the peak positions
pkAmp = nan(param.Np,time_points);          % store the peak amplitude
placeFlag = nan(param.Np,time_points);      % determine whether a place field

pkCenterMass = nan(param.Np,time_points);    % store the center of mass
pkMas = nan(param.Np,time_points);           % store the average ampltude 


ampThd = 0.1;               % amplitude threshold, depending on the parameters

% testing data, only used when check the representations
Y0 = zeros(param.Np,size(gdInput,2));
Yt = nan(param.Np,param.ps,time_points);


    
if strcmp(learnType, 'snsm')
    
    ystart = zeros(param.Np,BatchSize);  % inital of the output
    
    % generate position code by the grid cells    
    for i = 1:tot_iter
        positions = gdInput(:,randperm(param.ps,BatchSize));
        states = PlaceCellhelper.nsmDynBatch(positions,ystart, param);
        y = states.Y;
        
        % noisy update weight matrix
        param.W = (1-param.learnRate)*param.W + param.learnRate*y*positions'/BatchSize + ...
            sqrt(param.learnRate)*param.noise*randn(param.Np,param.Ng); 
        param.M = max(0,(1-param.learnRate)*param.M + param.learnRate*y*y'/BatchSize + ...
            sqrt(param.learnRate)*param.noise*randn(param.Np,param.Np));
        param.b = (1-param.learnRate)*param.b + param.learnRate*sqrt(param.alpha)*mean(y,2);
        
        % store and check representations
        Y0 = zeros(param.Np,size(gdInput,2));
        if mod(i, step) == 0
            states_fixed = PlaceCellhelper.nsmDynBatch(gdInput,Y0, param);
            flag = sum(states_fixed.Y > ampThd,2) > 3;  % only those neurons that have multiple non-zeros
            [~,pkInx] = sort(states_fixed.Y,2, 'descend');
            temp = states_fixed.Y((pkInx(:,1)-1)*param.Np + (1:param.Np)');
            pkAmp(flag,round(i/step)) = temp(flag);
            pks(flag,round(i/step)) = pkInx(flag,1);
            Yt(:,:,round(i/step)) = states_fixed.Y;
            
            % store the center of mass
            [pkCM, aveMass] = PlaceCellhelper.centerMassPks1D(states_fixed.Y,ampThd);
            pkCenterMass(:,round(i/step)) = pkCM;
            pkMas(:,round(i/step)) = aveMass;

        end

    end

elseif strcmp(learnType, 'inputNoise')
    
    ystart = 0.1*rand(param.Np,BatchSize);  % inital of the output
    
    posiInfo = nan(tot_iter,1);
%     ix = randperm(param.ps,1);   % random select seed
    
    % generate position code by the grid cells    
    for i = 1:tot_iter
        ix = PlaceCellhelper.nextPosi1D(ix,param);
        posiInfo(i) = ix;
        positions = gdInput(:,ix) + param.noise*randn(param.Ng,1);
        
        states = PlaceCellhelper.nsmDynBatch(positions,ystart, param);
        y = states.Y;
        
        % noisy update weight matrix
        param.W = (1-param.learnRate)*param.W + param.learnRate*y*positions'/BatchSize;
        param.M = (1-param.learnRate)*param.M + param.learnRate*y*y'/BatchSize;
        param.b = (1-param.learnRate)*param.b + param.learnRate*sqrt(param.alpha)*mean(y,2);
        
        % store and check representations
        Y0 = zeros(param.Np,size(gdInput,2));
        if mod(i, step) == 0
            states_fixed = PlaceCellhelper.nsmDynBatch(gdInput,Y0, param);
            flag = sum(states_fixed.Y > ampThd,2) > 3;  % only those neurons that have multiple non-zeros
            [~,pkInx] = sort(states_fixed.Y,2, 'descend');
            temp = states_fixed.Y((pkInx(:,1)-1)*param.Np + (1:param.Np)');
            pkAmp(flag,round(i/step)) = temp(flag);
            pks(flag,round(i/step)) = pkInx(flag,1);
            
             % store the center of mass
            [pkCM, mass] = PlaceCellhelper.centerMassPks1D(states_fixed.Y, ampThd);
            pkCenterMass(:,round(i/step)) = pkCM;
            pkMas(:,round(i/step)) = mass;
        end

    end
elseif strcmp(learnType, 'randwalk')
    
    ystart = zeros(param.Np,BatchSize);  % inital of the output
    
    posiInfo = nan(tot_iter,1);
    ix = randperm(param.ps,1);   % random select seed
    
    % generate position code by the grid cells    
    for i = 1:tot_iter
        ix = PlaceCellhelper.nextPosi1D(ix,param);
        posiInfo(i) = ix;
        positions = gdInput(:,ix);
        
        states = PlaceCellhelper.nsmDynBatch(positions,ystart, param);
        y = states.Y;
        
        % noisy update weight matrix
        param.W = (1-param.learnRate)*param.W + param.learnRate*y*positions'/BatchSize + ...
            sqrt(param.learnRate)*param.noise*randn(param.Np,param.Ng);  % 9/16/2020
        param.M = (1-param.learnRate)*param.M + param.learnRate*y*y'/BatchSize;
        param.b = (1-param.learnRate)*param.b + param.learnRate*sqrt(param.alpha)*mean(y,2);
        
        % store and check representations
        % store and check representations
        Y0 = zeros(param.Np,size(gdInput,2));
        if mod(i, step) == 0
            states_fixed = PlaceCellhelper.nsmDynBatch(gdInput,Y0, param);
            flag = sum(states_fixed.Y > ampThd,2) > 3;  % only those neurons that have multiple non-zeros
            [~,pkInx] = sort(states_fixed.Y,2, 'descend');
            temp = states_fixed.Y((pkInx(:,1)-1)*param.Np + (1:param.Np)');
            pkAmp(flag,round(i/step)) = temp(flag);
            pks(flag,round(i/step)) = pkInx(flag,1);
                       
             % store the center of mass
            [pkCM, mass] = PlaceCellhelper.centerMassPks(states_fixed.Y, ampThd);
            pkCenterMass(:,round(i/step)) = pkCM;
            pkMas(:,round(i/step)) = mass;
        end

    end
    
end


%% input and output similarity when random sampling from certain distribution

% output
ys = states_fixed.Y;
[~,iy] = sort(ys,2,'descend');
[~,in] = sort(iy(:,1),'ascend');

% plot heat map of final output
figure
imagesc(ys(in,:),[0,2])
colorbar
xlabel('Position')
ylabel('Neuron')


figure
imagesc(ys'*ys,[0,12])
colorbar
xlabel('Position')
ylabel('Position')

%% Estimate the diffusion constant

% estimate the mean squrare displacement
msds = nan(floor(time_points/2),size(pkMas,1));
for i = 1:size(msds,1)
    
    if param.Np == 1
        t1 = abs(pkCenterMass(:,i+1:end)' - pkCenterMass(:,1:end-i)');
    else
        t1 = abs(pkCenterMass(:,i+1:end) - pkCenterMass(:,1:end-i));
    end
    
    dx = min(t1,param.ps - t1);
    msds(i,:) = nanmean(dx.^2,2);
end


% linear regression to get the diffusion constant of each neuron
Ds = PlaceCellhelper.fitLinearDiffusion(msds,step,'linear');

%% Peak amplitdues

% shift of peak positions
epInx = randperm(param.Np,1);  % randomly slect
epPosi = [floor(pks(epInx,:)/param.ps);mod(pks(epInx,:),param.ps)]+randn(2,size(pks,2))*0.1;

% trajectory of peak
figure
plot((1:size(pkAmp,2))*step,pkAmp(epInx,:),'LineWidth',2)
xlabel('Time','FontSize',24)
ylabel('Peak amplitute','FontSize',24)
set(gca,'FontSize',24,'LineWidth',1.5)

% distribution of amplitude
figure
histogram(pkAmp(epInx,:))
xlabel('Peak amplitude','FontSize',24)
ylabel('Count','FontSize',24)
set(gca,'FontSize',24,'LineWidth',1.5)

% heatmap of peaks
figure
imagesc(pkAmp,[0,3])
colorbar
set(gca,'FontSize',20,'LineWidth',1.5)
xlabel('time','FontSize',24)
ylabel('neuron','FontSize',24)

% time period that a neuron has active place field
temp = pkAmp > ampThd;
tolActiTime = sum(temp,2)/size(pkAmp,2);
avePks = nan(size(pkAmp));
avePks(temp) = pkAmp(temp);
meanPks = nanmean(avePks,2);

figure
plot(meanPks,tolActiTime,'o','MarkerSize',8,'LineWidth',2)
xlabel('Average peak amplitude','FontSize',labelFont)
ylabel({'faction of', 'active time'},'FontSize',labelFont)
set(gca,'FontSize',axisFont,'LineWidth',axisLineWd)

figure
histogram(tolActiTime,20)
xlabel('active time fraction','FontSize',24)
ylabel('count','FontSize',24)
set(gca,'FontSize',20,'LineWidth',1.5)

%% Correlation of shift of centroids

%newPeaks = mydata;
trackL = 200;                 % length of the track
centroid_sep = 0:10:trackL;   % bins of centroid positions
deltaTau = 30;                % select time step seperation, subsampling
repeats = min(20,deltaTau);   % repeats

shiftCentroid = cell(length(centroid_sep)-1,repeats);
aveShift = nan(length(centroid_sep)-1,repeats);

for rp = 1:repeats
    initialTime = randperm(deltaTau,1);
    
    % a new method to estimate the correlations
    for bin = 1:length(centroid_sep)-1
        firstCen = [];
        secondCen = [];

        for i = initialTime:deltaTau:(size(pkCenterMass,2)-deltaTau)

            bothActiInx = ~isnan(pkCenterMass(:,i+deltaTau)) & ~isnan(pkCenterMass(:,i));
            ds = pkCenterMass(bothActiInx,i+deltaTau) - pkCenterMass(bothActiInx,i);
            temp = pkCenterMass(bothActiInx,i);
            
            % remove two ends
            rmInx = temp < 0.25*trackL | temp > 0.75*trackL;
            temp(rmInx) = nan;
            
            % pairwise active centroid of these two time points
            centroidDist = squareform(pdist(temp));
            [ix,iy] = find(centroidDist > centroid_sep(bin) & centroidDist <= centroid_sep(bin+1));
            firstCen = [firstCen;ds(ix)];
            secondCen = [secondCen;ds(iy)];
        end
        % merge
        shiftCentroid{bin,rp} = firstCen.*secondCen;
    %     aveShift(bin) = nanmean(firstCen.*secondCen)/nanstd(firstCen)/nanstd(secondCen);
        C = corrcoef(firstCen,secondCen);
        aveShift(bin,rp) = C(1,2);
    end
end
% avearge
aveRho = nan(length(centroid_sep)-1,repeats);

for rp = 1:repeats
    for i = 1:length(centroid_sep)-1
        aveRho(i,rp) = nanmean(shiftCentroid{i,rp});
    end
end
finalAveRho = nanmean(aveRho,2);
% plot the figure
figure
hold on
plot(centroid_sep(2:end)'/trackL,aveRho,'Color',greys(7,:), 'LineWidth',2)
plot(centroid_sep(2:end)'/trackL,finalAveRho,'Color',blues(7,:), 'LineWidth',4)
hold off
box on
xlabel('Distance (L)')
ylabel('$\langle \Delta r_A \Delta r_B\rangle$','Interpreter','latex')
title(['$\Delta t = ',num2str(deltaTau),'$'],'Interpreter','latex')

% Model centroid shift distribution
modelShift = [];
for rp = 1:repeats
    initialTime = randperm(deltaTau,1);
    for i = initialTime:deltaTau:(size(pkCenterMass,2)-deltaTau)

        bothActiInx = ~isnan(pkCenterMass(:,i+deltaTau)) & ~isnan(pkCenterMass(:,i));
        ds = pkCenterMass(bothActiInx,i+deltaTau) - pkCenterMass(bothActiInx,i);
        modelShift = [modelShift;ds];
    end
end 

figure
histogram(modelShift/trackL,'Normalization','pdf')
xlabel('Centroid shift (L)')
ylabel('pdf')


%% compared with pure random walk

wkspeed = 1;   % walk speed
numNeu = 100;
repeats = 20;  % repeat
initialPosi = randi(trackL,numNeu,1);  % initial centroid poistions
totSteps = round(size(pkCenterMass,2)/deltaTau);
nModelSample = length(modelShift);

% centroids_randomWalk = cell(repeats,1);  % st
shiftCentroid = cell(length(centroid_sep)-1,repeats);
aveShiftM = nan(length(centroid_sep)-1,repeats);

for rp = 1:repeats
    centroids_randomWalk = nan(numNeu,totSteps);
    for i = 1:numNeu
        temp = randi(trackL,1);
        ds = randperm(length(modelShift),1);  % random generate a shift
        for j = 1:totSteps
            newTry = temp + modelShift(randperm(nModelSample,1));
            if newTry < 0
                temp = abs(newTry);
            elseif newTry > trackL
                temp = max(1,2*trackL - newTry);  % can't be negative
            else
                temp = newTry;
            end             
            centroids_randomWalk(i,j) = temp;
        end
    end
    
    % distance dependent correlation
    for bin = 1:length(centroid_sep)-1
        firstCen = [];
        secondCen = [];

        for i = 1:totSteps-1

            bothActiInx = ~isnan(centroids_randomWalk(:,i+1)) & ~isnan(centroids_randomWalk(:,i));
            ds = centroids_randomWalk(bothActiInx,i+1) - centroids_randomWalk(bothActiInx,i);
            temp = centroids_randomWalk(bothActiInx,i);
            
            % remove two ends
            rmInx = temp < 0.25*trackL | temp > 0.75*trackL;
            temp(rmInx) = nan;
            
            % pairwise active centroid of these two time points
            centroidDist = squareform(pdist(temp));
            [ix,iy] = find(centroidDist > centroid_sep(bin) & centroidDist <= centroid_sep(bin+1));
            firstCen = [firstCen;ds(ix)];
            secondCen = [secondCen;ds(iy)];
        end
        % merge
        shiftCentroid{bin,rp} = firstCen.*secondCen;
        C = corrcoef(firstCen,secondCen);
        aveShiftM(bin,rp) = C(1,2);
    end
end


figure
hold on
fh = shadedErrorBar((1:size(aveShift,1))',aveShift',{@mean,@std});
box on
set(fh.edge,'Visible','off')
fh.mainLine.LineWidth = 4;
fh.mainLine.Color = blues(10,:);
fh.patch.FaceColor = blues(7,:);

fh2 = shadedErrorBar((1:size(aveShiftM,1))',aveShiftM',{@mean,@std});
box on
set(fh2.edge,'Visible','off')
fh2.mainLine.LineWidth = 3;
fh2.mainLine.Color = greys(7,:);
fh2.patch.FaceColor = greys(5,:);
hold off
legend('Model','Random walk')
set(gca,'XTick',0:5:10,'XTickLabel',{'0','0.25','0.5'})
title(['$\Delta t = ',num2str(deltaTau),'$'],'Interpreter','latex')
xlabel('Distance (L)')
ylabel('$\rho$','Interpreter','latex')
