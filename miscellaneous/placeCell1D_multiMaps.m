% examing how representational drift change if multiple maps are learned
% sequentially
% using non-negative similarity matching to learn a 1-d place cells 
% input is grid cell response vectors
% 1D grid fields are slices of 2D lattice of grid fields

% add input noise simulation
% last revised  9/2/2021
clear
close all

%% model parameters
param.ps =  200;        % number of positions along each dimension
% param.Nlbd = 5;       % number of different scales/spacing,default 5
param.Nlbd = 5;         % number of different scales/spacing
param.Nthe = 6;         % number of rotations
param.Nx =  4;          % offset of x-direction
param.Ny = 4;           % offset of y-direction
param.Ng = param.Nlbd*param.Nthe*param.Nx*param.Ny;   % total number of grid cells
param.Np = 200;       % number of place cells, default 20*20
param.env = 4;        % number of environments to be learned

param.baseLbd = 1/4;  % spacing of smallest grid RF, default 1/4
param.sf =  1.42;     % scaling factor between adjacent module

% parameters for learning 
noiseStd =0.01;          % 0.001
learnRate = 0.02;       % default 0.05

param.W = 0.5*randn(param.Np,param.Ng);   % initialize the forward matrix
param.M = eye(param.Np);        % lateral connection if using simple nsm
param.lbd1 = 0.00;              % 0.15 for 400 place cells and 5 modes
param.lbd2 = 0.1;              % 0.05 for 400 place cells and 5 modes


param.alpha = 60;   % the threshold depends on the input dimension, 65
param.beta = 2; 
param.gy = 0.05;   % update step for y
param.gz = 0.1;    % update step for z
param.gv = 0.2;    % update step for V
param.b = zeros(param.Np,1);  % biase
param.learnRate = learnRate;  % learning rate for W and b
param.noise =  noiseStd;   % stanard deivation of noise 
param.rwSpeed = 4;         % steps each update, default 1

param.ori = 1/10*pi;     % slicing orientation 1
param.ori2 = 1/6*pi;  

BatchSize = 1;      % minibatch used to to learn
learnType = 'snsm';  % snsm, batch, online, randwalk, direction, inputNoise

gridQuality = 'slice';  % regular, weak or slice


makeAnimation = 0;    % whether make a animation or not

% generate grid fields 
% 1D slice of the 2D grid fields
% ori = 1/12*pi;    % slicing direction
posiGram = eye(param.ps);
allGridFields = nan(param.ps,param.Ng,param.env);  % store all the grid fields
gdInputAll = nan(param.Ng,param.ps,param.env);   % 
allGridFields(:,:,1) = slice2Dgrid(param.ps,param.Nlbd,param.Nthe,param.Nx,param.Ny,param.ori);
gdInputAll(:,:,1) = allGridFields(:,:,1)'*posiGram;


allGridFields(:,:,1) = slice2Dgrid(param.ps,param.Nlbd,param.Nthe,param.Nx,param.Ny,param.ori);
gdInputAll(:,:,1) = allGridFields(:,:,1)'*posiGram;


for i = 2:param.env
gridFields = slice2Dgrid(param.ps,param.Nlbd,param.Nthe,param.Nx,param.Ny,param.ori);
    allGridFields(:,:,i) = slice2Dgrid(param.ps,param.Nlbd,param.Nthe,param.Nx,param.Ny,param.ori2,"shifted");
    gdInputAll(:,:,i)= allGridFields(:,:,i)'*posiGram;  % gramin of input matrix
end


% examing these two grid cell inputs
% figure
% subplot(1,2,1)
% imagesc(gridFields)
% 
% subplot(1,2,2)
% imagesc(gridFields2)
% 
% figure
% imagesc(gridFields - gridFields2)
% % 
% figure
% hold on
% plot(gridFields(:,100))
% plot(gridFields2(:,100))
% % 
% figure
% imagesc(gridFields*gridFields',[50,90])
% % 
% figure
% imagesc(gridFields2*gridFields2')
%% using non-negative similarity matching to learn place fields
% generate input from grid filds

tot_iter = 1e4;   % total interation, default 2e3
sep = 20;

posiGram = eye(param.ps);
% gdInput = gridFields'*posiGram;  % gramin of input matrix
% gdInputTot = [gridFields'*posiGram,gridFields2'*posiGram];  % gramin of input matrix
gdInputTot = reshape(gdInputAll,[param.Ng,param.ps*param.env]);

% gdInput = gridFields'*posiGram*3/sqrt(param.Ng);  % gramin of input matrix

if strcmp(learnType, 'online')
    for i = 1:1000
        x = X(:,randperm(t,1));  % randomly select one input
        [states, param] = PlaceCellhelper.neuralDynOnline(x,Y0,Z0,V, param);
        y = states.y;
        % update weight matrix
        param.W = (1-param.learnRate)*param.W + y*x';
        param.b = (1-param.learnRate)*param.b + param.learnRatessqrt(param.alpha)*y;
        V = states.V;   % current 
    end
elseif strcmp(learnType, 'batch')
    for i = 1:tot_iter
        x = gdInput(:,randperm(param.ps*param.ps,BatchSize));
%         x = X(:,randperm(param.ps*param.ps,BatchSize));  % randomly select one input
        [states, param] = PlaceCellhelper.neuralDynBatch(x,Y0,Z0,V, param);
        y = states.Y;
        % update weight matrix
        param.W = (1-param.learnRate)*param.W + param.learnRate*y*x'/BatchSize;
        param.b = (1-param.learnRate)*param.b + param.learnRate*sqrt(param.alpha)*mean(y,2);
        V = states.V;   % current 
    end
% simple non-negative similarity matching
elseif strcmp(learnType, 'snsm')
    ystart = zeros(param.Np,BatchSize);  % inital of the output
    
    % generate position code by the grid cells
%     posiInfo = nan(tot_iter,1);
    totPosi = param.ps*param.env;
    for i = 1:tot_iter
%         positions = gdInput(:,randperm(param.ps,BatchSize));
%         positions = gdInputTot(:,randperm(param.ps,BatchSize));
        positions = gdInputTot(:,randperm(totPosi,BatchSize));
%         posi_sel = ceil(betarnd(0.5,0.5,BatchSize,1)*param.ps); % oversampling at the two ends
%         positions = gdInput(:,posi_sel);
%         y = PlaceCellhelper.quadprogamYfixed(positions,param);
        states= PlaceCellhelper.nsmDynBatch(positions,ystart, param);
        y = states.Y;
        
%         posiInfo(i) = posi_sel;
        
        
        % update weight matrix
%         param.W = (1-param.learnRate)*param.W + param.learnRate*y*positions'/BatchSize;
%         param.M = (1-param.learnRate)*param.M + param.learnRate*y*y'/BatchSize;
        param.W = (1-param.learnRate)*param.W + param.learnRate*y*positions'/BatchSize + ...
            sqrt(param.learnRate)*param.noise*randn(param.Np,param.Ng);
        param.M = max(0,(1-param.learnRate)*param.M + param.learnRate*y*y'/BatchSize + ...
            sqrt(param.learnRate)*param.noise*randn(param.Np,param.Np));
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
%         ix = PlaceCellhelper.nextPosi1D(ix,param);
        ix = max(1,mod(i,param.ps));
        posiInfo(i) = ix;
        positions = gdInput(:,ix) + param.noise*randn(param.Ng,1);  % column-wise storage, add Gaussian noise
%         positions = gdInput(:,randperm(param.ps*param.ps,1));
        
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


% ys = nan(param.Np,numProb);
% for i = 1:numProb
%     ys(:,i) = PlaceCellhelper.quadprogamYfixed(gdInput(:,i),param);
% end
gdInput = allGridFields(:,:,1)'*posiGram;        % gramin of input matrix
ys_prob = zeros(param.Np,numProb);
states= PlaceCellhelper.nsmDynBatch(gdInput,ys_prob, param);
ys = states.Y;

% representation of the new environment
gdInput2 = allGridFields(:,:,end)'*posiGram;        % gramin of input matrix
states2= PlaceCellhelper.nsmDynBatch(gdInput2,ys_prob, param);
ys2 = states2.Y;


figure
imagesc(ys)

figure
imagesc(ys2)

% figure
% imagesc(param.W*(gdInput2 - gdInput),[0,2])
% 
% figure
% imagesc(gdInput2 - gdInput)
% figure
% hold on
% plot(ys(100,:))
% plot(ys2(100,:))

% compare the position of centroid
[~,pkInx] = sort(ys,2, 'descend');
% pkAmp(:,round(i/step)) = ys((pkInx(:,1)-1)*param.Np + (1:param.Np)');
pkAmp1= ys((pkInx(:,1)-1)*param.Np + (1:param.Np)');
pks1 = nan(param.Np,1);
% pks1 = pkInx(:,1);

pks1(pkAmp1 > 0.1) = pkInx(pkAmp1 > 0.1,1);



[~,pkInx2] = sort(ys2,2, 'descend');
pkAmp2= ys2((pkInx2(:,1)-1)*param.Np + (1:param.Np)');
pks2 = nan(param.Np,1);

pks2(pkAmp2 > 0.1) = pkInx2(pkAmp2 > 0.1,1);

figure
plot(pks1,pks2,'o')

% pcInx = pkAmp(:,round(i/step)) > ampThd;  % find the place cells
% pks(pcInx,round(i/step)) = pkInx(pcInx,1);
figure
imagesc(ys'*ys)

figure
plot(ys(99,:))

%% Analysis and plot
%{
% y = allY(:,:,end);
% =============== Place field  of all the place cells ===============
% pfs = y*posiGram'./(sum(y,2)*ones(1,param.ps));
% figure
% plot(pfs(randperm(param.Np,5),:)')


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
%}

%% Continuous Nosiy update
% 
tot_iter = 1e4;
num_sel = 100;
step = 10;
time_points = round(tot_iter/step);

% allW = nan(tot_iter/sep,size(gridFields,2));
% allbias = nan(tot_iter/sep,1);

pks = nan(param.Np,time_points);    % store all the peak positions
pkAmp = nan(param.Np,time_points);  % store the peak amplitude
placeFlag = nan(param.Np,time_points); % determine whether a place field

pkCenterMass = nan(param.Np,time_points);  % store the center of mass
pkMas = nan(param.Np,time_points);           % store the average ampltude 


ampThd = 0.1;   % amplitude threshold, depending on the parameters

% testing data, only used when check the representations
% Xsel = gdInput(:,1:10:end);
Y0 = zeros(param.Np,size(gdInput,2));
% Z0 = zeros(numIN,size(gdInput,2));
Yt = nan(param.Np,param.ps,time_points);
if strcmp(learnType, 'online')
    for i = 1:tot_iter
        x = X(:,randperm(t,1));  % randomly select one input
        [states, params] = MantHelper.neuralDynOnline(x,Y0,Z0,V, params);
        y = states.y;
        % update weight matrix
        params.W = (1-params.learnRate)*params.W + params.learnRate*y*x' + ...
            sqrt(params.learnRate)*params.noise*randn(k,2);
        params.b = (1-params.learnRate)*params.b + params.learnRate*sqrt(params.alpha)*y;
        V = states.V;   % current feedback matrix
        
    end
elseif strcmp(learnType, 'batch')
    y0 = zeros(param.Np, BatchSize);  % initialize mini-batch
    z0 = zeros(numIN,BatchSize);  % initialize interneurons

    for i = 1:tot_iter
        x = gdInput(:,randperm(param.ps*param.ps,BatchSize));
%         x = X(:,randperm(param.ps*param.ps,BatchSize));  % randomly select one input
        [states, param] = PlaceCellhelper.neuralDynBatch(x,y0,z0,V, param);
        y = states.Y;
        % update weight matrix
        param.W = (1-param.learnRate)*param.W + param.learnRate*y*x'/BatchSize...
            + sqrt(param.learnRate)*param.noise*randn(param.Np,param.Ng);
        param.b = (1-param.learnRate)*param.b + param.learnRate*sqrt(param.alpha)*mean(y,2);
        V = states.V;   % current 
        
        if mod(i, step) == 0
            [states_fixed,param] = PlaceCellhelper.neuralDynBatch(gdInput,Y0,Z0,V, param);
            [~,pkInx] = sort(states_fixed.Y,2, 'descend');
            pkAmp(:,round(i/step)) = states_fixed.Y((pkInx(:,1)-1)*param.Np + (1:param.Np)');
            pcInx = pkAmp(:,round(i/step)) > ampThd;  % find the place cells
            pks(pcInx,round(i/step)) = pkInx(pcInx,1);
%             Yt(:,:,round(i/step)) = states_fixed.Y;

        end
    end
elseif strcmp(learnType, 'snsm')
    
    ystart = zeros(param.Np,BatchSize);  % inital of the output
   
    % generate position code by the grid cells    
    for i = 1:tot_iter
%         positions = gdInput(:,randperm(param.ps,BatchSize));
        positions = gdInputTot(:,randperm(totPosi,BatchSize));
%         posi_sel = ceil(betarnd(2,5,BatchSize,1)*param.ps); % oversampling at the two ends
%         positions = gdInput(:,posi_sel);
        states = PlaceCellhelper.nsmDynBatch(positions,ystart, param);
        y = states.Y;
        
        % noisy update weight matrix
        param.W = (1-param.learnRate)*param.W + param.learnRate*y*positions'/BatchSize + ...
            sqrt(param.learnRate)*param.noise*randn(param.Np,param.Ng);  % 9/16/2020
%         param.M = (1-param.learnRate)*param.M + param.learnRate*y*y'/BatchSize;
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
        
%         if mod(i,timeGap) ==0
%             allW(:,:,round(i/timeGap)) = param.W;
%             allM(:,:,round(i/timeGap)) = param.M;
%             allY(:,:,round(i/timeGap)) = states_fixed.Y;
%         end
               
    end

elseif strcmp(learnType, 'inputNoise')
    
    ystart = 0.1*rand(param.Np,BatchSize);  % inital of the output
    
    posiInfo = nan(tot_iter,1);
%     ix = randperm(param.ps,1);   % random select seed
    
    % generate position code by the grid cells    
    for i = 1:tot_iter
        ix = PlaceCellhelper.nextPosi1D(ix,param);
%         ix = max(1, mod(i,param.ps));
        posiInfo(i) = ix;
        positions = gdInput(:,ix) + param.noise*randn(param.Ng,1);
        
        states = PlaceCellhelper.nsmDynBatch(positions,ystart, param);
        y = states.Y;
        
        % noisy update weight matrix
        param.W = (1-param.learnRate)*param.W + param.learnRate*y*positions'/BatchSize;
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
%             pcInx = pkAmp(:,round(i/step)) > ampThd & flag;  % find the place cells
            pks(flag,round(i/step)) = pkInx(flag,1);
%             Yt(:,:,round(i/step)) = states_fixed.Y;
                       
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
%             pcInx = pkAmp(:,round(i/step)) > ampThd & flag;  % find the place cells
            pks(flag,round(i/step)) = pkInx(flag,1);
%             Yt(:,:,round(i/step)) = states_fixed.Y;
                       
             % store the center of mass
            [pkCM, mass] = PlaceCellhelper.centerMassPks(states_fixed.Y, ampThd);
            pkCenterMass(:,round(i/step)) = pkCM;
            pkMas(:,round(i/step)) = mass;
        end

    end
    
end

%% analyzing the drift behavior
% pca of the peak amplitude=  dynamcis
% [COEFF, SCORE, ~, ~, EXPLAINED] = pca(pkAmp');
% 
% 
% figure
% plot(cumsum(EXPLAINED),'LineWidth',3)
% xlabel('pc','FontSize',24)
% ylabel('cummulative variance','FontSize',24)
% set(gca,'FontSize',20,'LineWidth',1.5)


%% input and output similarity when random sampling from certain distribution
% Xsamp = gdInput(:,posiInfo);
% [~,ix] = sort(posiInfo,'ascend');
% Xsorted = Xsamp(:,ix(1:20:end));
% figure
% imagesc(Xsorted'*Xsorted)
% xlabel('position')
% ylabel('position')


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


%% Peak dynamics

% shift of peak positions
epInx = randperm(param.Np,1);  % randomly slect
epPosi = [floor(pks(epInx,:)/param.ps);mod(pks(epInx,:),param.ps)]+randn(2,size(pks,2))*0.1;

specColors = brewermap(size(epPosi,2), 'Spectral');

% colors indicate time
figure
hold on
for i=1:size(pks,2)
    plot(epPosi(1,i),epPosi(2,i),'^','MarkerSize',4,'MarkerEdgeColor',...
        specColors(i,:),'LineWidth',1.5)
end
hold off
box on
xlabel('x position','FontSize',24)
ylabel('y position','FontSize',24)
set(gca,'FontSize',24,'LineWidth',1.5)

% 2d trajectory
figure
plot(epPosi(1,:),epPosi(2,:), 'o-','MarkerSize',4,'LineWidth',2)
xlabel('x position','FontSize',24)
ylabel('y position','FontSize',24)
set(gca,'FontSize',24,'LineWidth',1.5)

% distribution of step size
ixs = epPosi(1,:)>0;
times =1:size(epPosi,2);
pkTimePoints = times(ixs);
silentInt = diff(pkTimePoints)-1; % silent interval
figure
histogram(silentInt(silentInt > 0)*step)
xlabel('$\Delta t$','Interpreter','latex','FontSize',24)
ylabel('count','FontSize',24)
set(gca,'FontSize',24,'LineWidth',1.5)


temp = epPosi(:,ixs);
dp = [diff(temp(1,:));diff(temp(2,:))];
ds = sqrt(sum(dp.^2,1));

% distribution of
figure
histogram(log(ds(ds>=1)),32)
xlabel('$\ln(\Delta d)$','Interpreter','latex','FontSize',24)
ylabel('count','FontSize',24)
set(gca,'FontSize',24,'LineWidth',1.5)

[f,X] = ecdf(ds(ds>=1 & ds < 20));
figure
plot(X,1 - f,'LineWidth',3)
xlabel('$\log_{10}(\Delta d)$','Interpreter','latex','FontSize',24)
ylabel('$1-F(x)$','Interpreter','latex','FontSize',24)
set(gca,'XScale','log','YScale','log','LineWidth',1.5,'FontSize',24)


% statistics of jump and local diffusion
ds = diff(epPosi,1,2);
dsTraj = sqrt(sum(ds.^2,1))/param.ps;
figure
plot(dsTraj,'LineWidth',1)
xlabel('time','FontSize',24)
ylabel('$\Delta d$','Interpreter','latex','FontSize',24)
set(gca,'LineWidth',1.5,'FontSize',20)

% make an animation of peak positions
% randomly select 10 neurons
selNeur = randperm(param.Np,min(10,param.Np));
set1 = brewermap(11,'Set1');
frameSep = 5; % only capture every 5 steps, each step  = 2 iterations

epPosiXs = floor(pks(selNeur,:)/param.ps);
epPosiYs = mod(pks(selNeur,:),param.ps); 


if makeAnimation
    figure
    set(gcf,'color','w');
    hold on
    for i = 1:length(selNeur)
        plot(epPosiXs(i,1),epPosiYs(i,1),'o','MarkerSize',20,'MarkerEdgeColor',...
            set1(i,:),'LineWidth',3)
        xlim([0,32])
        ylim([0,32])
        text(epPosiXs(i,1)-0.3,epPosiYs(i,1),num2str(i),'FontSize',20)
    end
    hold off
    box on
    xlabel('x','FontSize',20)
    ylabel('y','FontSize',20)
    set(gca,'LineWidth',1.5,'FontSize',20)

    detlaT = 0.05;
    f = getframe(gcf);
    [im,map] = rgb2ind(f.cdata,256,'nodither');
    k = 1;
    for i = 1:round(size(epPosiXs,2)/frameSep)
        for j=1:length(selNeur)
            plot(epPosiXs(j,i*frameSep),epPosiYs(j,i*frameSep),'o','MarkerSize',20,'MarkerEdgeColor',...
            set1(j,:),'LineWidth',3)
            text(epPosiXs(j,i*frameSep)-0.3,epPosiYs(j,i*frameSep),num2str(j),'FontSize',16)
            hold on
        end
        hold off
        box on
        xlim([0,32])
        ylim([0,32])
        xlabel('x','FontSize',20)
        ylabel('y','FontSize',20)
        set(gca,'LineWidth',1.5,'FontSize',20)
        title(['time = ', num2str(i*frameSep)])
        f = getframe(gcf);
        im(:,:,1,k) = rgb2ind(f.cdata,map,'nodither');
        k = k + 1;

    end
    imwrite(im,map,'scatterPkPosi.gif','DelayTime',detlaT,'LoopCount',inf)
end
%% Estimate the diffusion constant

% estimate the msd
% msds = nan(min(floor(time_points/2),1000),size(pkMas,1));
msds = nan(floor(time_points/2),size(pkMas,1));
for i = 1:size(msds,1)
    
%     xcoord = pkCenterMass(i,:);
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

%% interval of active, silent, shift of RF

% interval length where place fields disappears
ampThd = 0.1;
pkFlags = pkAmp > ampThd;
silentInter = [];  % store the silent interval
activeInter = [];
randActiInter = [];  % compare with random case
actInxPerm = reshape(pkFlags(randperm(size(pkFlags,1)*size(pkFlags,2))),size(pkFlags));
for i = 1:size(pkAmp,1)
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


% correlation between RF shift and total time that are silent


%% Correlation of shift of centroids

%newPeaks = mydata;
trackL = 200;      % length of the track
centroid_sep = 0:10:trackL;   % bins of centroid positions
deltaTau = 100;      % select time step seperation, subsampling
% numAnimals = size(pc_Centroid.allCentroids,1);
repeats = min(20,deltaTau);

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
            rmInx = temp < 0.2*trackL | temp > 0.8*trackL;
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
%save the figures
figPref = [sFolder,filesep,['1D_place_shift_corr_raw_rm',num2str(deltaTau)]];
saveas(gcf,[figPref,'.fig'])
print('-depsc',[figPref,'.eps'])



% figPref = [sFolder,filesep,['1D_place_shift_corr_normalized_rm',num2str(deltaTau)]];
% saveas(gcf,[figPref,'.fig'])
% print('-depsc',[figPref,'.eps'])


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

figPref = [sFolder,filesep,['1D_place_shift_hist_',num2str(deltaTau)]];
saveas(gcf,[figPref,'.fig'])
print('-depsc',[figPref,'.eps'])






std(modelShift)


figure
plot(nanmean(aveShift,2))

%% compared with pure random walk

wkspeed = 1;   % walk speed
numNeu = 200;
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
            rmInx = temp < 0.2*trackL | temp > 0.8*trackL;
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
% plot(centroid_sep(2:end)',aveShift,'Color',greys(7,:),'LineWidth',2)
% plot(centroid_sep(2:end)',nanmean(aveShift,2),'Color',blues(7,:),'LineWidth',4)
% hold off
% box on
legend('Model','Random walk')
set(gca,'XTick',0:5:10,'XTickLabel',{'0','0.25','0.5'})
title(['$\Delta t = ',num2str(deltaTau),'$'],'Interpreter','latex')
xlabel('Distance (L)')
ylabel('$\rho$','Interpreter','latex')

figPref = [sFolder,filesep,['1Dslice_corr_rm',num2str(deltaTau)]];
saveas(gcf,[figPref,'.fig'])
print('-depsc',[figPref,'.eps'])


% save the average correlations for plot in "hippo_Gonzalez_figures.m'
save('./data/1D_slice_centroidCorr_0708.mat','aveShift','aveShiftM')

%% Publication ready figures

% this part polish some of the figures and make them publication ready
% define all the colors
sFolder = ['.',filesep,'figures'];
figPre = ['placeCell_1D_slice_N200_',date];

nc = 256;   % number of colors
spectralMap = brewermap(nc,'Spectral');
PRGnlMap = brewermap(nc,'PRGn');
% RdBuMap = flip(brewermap(nc,'RdBu'),1);

blues = brewermap(11,'Blues');
reds = brewermap(11,'Reds');
BuGns = brewermap(11,'BuGn');
greens = brewermap(11,'Greens');
greys = brewermap(11,'Greys');
PuRd = brewermap(11,'PuRd');
YlOrRd = brewermap(11,'YlOrRd');
oranges = brewermap(11,'Oranges');
RdBuMap = flip(brewermap(nc,'RdBu'),1);
BlueMap = flip(brewermap(nc,'Blues'),1);
GreyMap = brewermap(nc,'Greys');


% Parameters for plot
figWidth = 3.5;   % in unit of inches
figHeight = 2.8;
plotLineWd = 2;    %line width of plot
axisLineWd = 1;    % axis line width
labelFont = 20;    % font size of labels
axisFont = 16;     % axis font size


% ************************************************************
% peak position, using time point 1
% ************************************************************
temp = [ceil(pks(:,1)/param.ps),mod(pks(:,1),param.ps)]/param.ps;
pkVec = temp(~isnan(temp(:,1)),:);

% define figure size
f_pkPosi = figure;
set(f_pkPosi,'color','w','Units','inches')
pos(3)=3.2;  %17.4; %8.5;%pos(3)*1.5;
pos(4)=2.8;  % pos(4)*1.5;
set(f_pkPosi,'Position',pos)
% histogram(activeInter(:))
% xlim([0,100])
plot(pkVec(:,1),pkVec(:,2),'o','MarkerSize',4,'MarkerFaceColor',greys(9,:),...
    'MarkerEdgeColor',greys(9,:))
xlabel('X position','FontSize',16)
ylabel('Y position','FontSize',16)
set(gca,'LineWidth',1,'FontSize',16)

prefix = [figPre, 'pkPosi'];
saveas(f_pkPosi,[sFolder,filesep,prefix,'.fig'])
print('-depsc',[sFolder,filesep,prefix,'.eps'])



% ************************************************************
% peak amplitude of example neuron
% ************************************************************
inx = 34;  % select one neuron to plot
temp = pkAmp(inx,:);
temp(isnan(temp)) = 0;
times = (1:length(temp))*step;   % iterations

f_pkAmpTraj = figure;
set(f_pkAmpTraj,'color','w','Units','inches')
pos(3)=3.2;  %17.4; %8.5;%pos(3)*1.5;
pos(4)=2.8;%pos(4)*1.5;
set(f_pkAmpTraj,'Position',pos)

plot(times',temp','LineWidth',2,'Color',blues(9,:))
xlabel('Time','FontSize',16)
ylabel('Peak Amplitude','FontSize',16)
set(gca,'LineWidth',1,'FontSize',16)

prefix = [figPre, 'pkAmp'];
saveas(f_pkAmpTraj,[sFolder,filesep,prefix,'.fig'])
print('-depsc',[sFolder,filesep,prefix,'.eps'])

% ************************************************************
% peak amplitude of all neurons across all time
% ************************************************************
f_pkAmpall = figure;
set(f_pkAmpall,'color','w','Units','inches')
pos(3)=3.2;  %17.4; %8.5;%pos(3)*1.5;
pos(4)=2.8;%pos(4)*1.5;
set(f_pkAmpall,'Position',pos)

pkAllData = pkAmp(~isnan(pkAmp));
histogram(pkAllData,'Normalization','pdf')
xlim([0,8])
xlabel('peak amplitude','FontSize',16)
ylabel('pdf','FontSize',16)
set(gca,'LineWidth',1,'FontSize',16)


% ************************************************************
% Fraction of active neurons
% ************************************************************
f_acti = figure;
set(f_acti,'color','w','Units','inches')
pos(3)=3.3;  
pos(4)=2.8;
set(f_acti,'Position',pos)
actiFrac = mean(pks > 0,1);

plot(times',actiFrac,'LineWidth',1.5)
xlabel('Time','FontSize',16)
ylabel('Active fraction','FontSize',16)
ylim([0,1])
set(gca,'LineWidth',1,'FontSize',16)

prefix = [figPre, 'fracActive'];
saveas(f_acti,[sFolder,filesep,prefix,'.fig'])
print('-depsc',[sFolder,filesep,prefix,'.eps'])


% ************************************************************
% distribution of active time
% ************************************************************
f_actiTime = figure;
set(f_actiTime,'color','w','Units','inches')
pos(3)=3.5;  
pos(4)=2.8;
set(f_actiTime,'Position',pos)
actiTime = mean(pks > 0,2);

histogram(actiTime,20);
xlabel('Active time','FontSize',16)
ylabel('Count','FontSize',16)
set(gca,'LineWidth',1,'FontSize',16)


% ************************************************************
% peak position of example neuron
% ************************************************************
epInx = randperm(param.Np,1);  % randomly slect
epPosi = [floor(pks(inx,:)/param.ps);mod(pks(inx,:),param.ps)]+randn(2,size(pks,2))*0.1;

specColors = brewermap(size(epPosi,2), 'Spectral');

% colors indicate time
f_pkPosi = figure;
set(f_pkPosi,'color','w','Units','inches')
pos(3)=3.2;  
pos(4)=2.8;
set(f_pkPosi,'Position',pos)
hold on

for i=1:size(pks,2)
    plot(epPosi(1,i),epPosi(2,i),'o','MarkerSize',4,'MarkerEdgeColor',...
        specColors(i,:),'LineWidth',1.5)
end
hold off
box on
xlabel('x position','FontSize',16)
ylabel('y position','FontSize',16)
set(gca,'FontSize',16,'LineWidth',1)

% ************************************************************
% average peak amplitude vs time of active
% ************************************************************

temp = pkAmp > ampThd;
tolActiTime = sum(temp,2)/size(pkAmp,2);
avePks = nan(size(pkAmp));
avePks(temp) = pkAmp(temp);
meanPks = nanmean(avePks,2);

f_ampActiTime= figure;
set(f_ampActiTime,'color','w','Units','inches')
pos(3)=3.5;  
pos(4)=2.8;
set(f_ampActiTime,'Position',pos)
plot(meanPks,tolActiTime,'o','MarkerSize',4,...
    'MarkerEdgeColor',greys(9,:),'LineWidth',1.5)
xlabel('Mean peak amplitude','FontSize',labelFont)
% ylabel({'faction of', 'active time'},'FontSize',labelFont)
ylabel(['faction of ', 'active time'],'FontSize',labelFont)
set(gca,'FontSize',axisFont,'LineWidth',axisLineWd)

prefix = [figPre, 'pkAmp_actiTime'];
saveas(f_ampActiTime,[sFolder,filesep,prefix,'.fig'])
print('-depsc',[sFolder,filesep,prefix,'.eps'])

% ************************************************************
% distribution of peak shift
% ************************************************************
% prepare the data
shiftStat = nan(param.Np,2);  %store the mean and std of peak shift
allShiftDist = [];
for i = 1:param.Np
    flag = find(pks(i,:) > 0);
    temp = pks(i,flag);
%     tsq = zeros(1,size(pks,2));
%     tsq(flag) = flag;
%     jumpFlag = find(diff(tsq) > 1) + 1;
    jumpFlag = find(diff(flag)>1);
    
    
    xs = floor(temp/param.ps);
    ys = mod(temp,param.ps);
    shiftFlag = find(diff(temp) ~= 0); % whether shift or not
    
    jointFlag = union(shiftFlag,jumpFlag);
    
    dxs = xs(jointFlag + 1) - xs(jointFlag);
    dys = ys(jointFlag + 1) - ys(jointFlag);
    shiftEcul = sqrt(dxs.^2 + dys.^2)/param.ps;
    shiftStat(i,:) = [mean(shiftEcul),std(shiftEcul)]; 
    allShiftDist = [allShiftDist,shiftEcul];
end

% active peak amplitude and peak shift
f_pkShift = figure;
set(f_pkShift,'color','w','Units','inches')
pos(3)=3.5;  
pos(4)=2.8;
set(f_pkShift,'Position',pos)
plot(meanPks,shiftStat(:,1),'o','MarkerSize',4,...
    'MarkerEdgeColor',greys(9,:),'LineWidth',1.5)
xlabel('Average peak amplitude','FontSize',16)
ylabel('$\langle \Delta r \rangle$','Interpreter','latex','FontSize',16)
set(gca,'FontSize',16,'LineWidth',1)
ylim([0,0.1])

prefix = [figPre, 'pkShit_actiTime'];
saveas(f_pkShift,[sFolder,filesep,prefix,'.fig'])
print('-depsc',[sFolder,filesep,prefix,'.eps'])


% active peak amplitude and 
% meanCM = nanmean(pkCenterMass,2);

diffContFig = figure;
pos(3)=3.5;  
pos(4)=2.8;
set(diffContFig,'color','w','Units','inches','Position',pos)
plot(meanPks,Ds(:,1),'o','MarkerSize',4,...
    'MarkerEdgeColor',greys(9,:),'LineWidth',1.5)
xlabel('Mean peak amplitude','FontSize',labelFont)
ylabel('$D$','Interpreter','latex','FontSize',labelFont)
set(gca,'FontSize',axisFont,'LineWidth',axisLineWd)
ylim([0,2.5])

prefix = [figPre, 'amp_diffCont_'];
saveas(diffContFig,[sFolder,filesep,prefix,'.fig'])
print('-depsc',[sFolder,filesep,prefix,'.eps'])


% Diffusion constant and active period
DiffuAmpFig = figure;
pos(3)=3.5;  
pos(4)=2.8;
set(DiffuAmpFig,'color','w','Units','inches','Position',pos)
plot(tolActiTime,Ds(:,1),'o','MarkerSize',4,...
    'MarkerEdgeColor',greys(9,:),'LineWidth',1.5)
xlabel('Fraction of active time','FontSize',labelFont)
ylabel('$D$','Interpreter','latex','FontSize',labelFont)
set(gca,'FontSize',axisFont,'LineWidth',axisLineWd)
ylim([0,2.5])

prefix = [figPre, 'active_diffCont_'];
saveas(DiffuAmpFig,[sFolder,filesep,prefix,'.fig'])
print('-depsc',[sFolder,filesep,prefix,'.eps'])


figure
ecdf(allShiftDist)
histogram(allShiftDist(allShiftDist>0.1))

% ************************************************************
% Change of population vectors, measured by correlation
% ************************************************************

% fmSel = randperm(size(Yt,2),1); % randomly slect three frames to consider
pvCorr = zeros(size(Yt,3),size(Yt,2)); 
% [~,neuroInx] = sort(peakInx(:,inxSel(1)));

for i = 1:size(Yt,3)
    for j = 1:size(Yt,2)
        temp = Yt(:,j,i);
        C = corrcoef(temp,Yt(:,j,1));
        pvCorr(i,j) = C(1,2);
    end
end

f_pvCorr = figure;
set(f_pvCorr,'color','w','Units','inches')
pos(3)=figWidth;  
pos(4)=figHeight;
set(f_pvCorr,'Position',pos)
fh = shadedErrorBar((1:size(pvCorr,1))',pvCorr',{@mean,@std});
box on
set(fh.edge,'Visible','off')
fh.mainLine.LineWidth = 3;
fh.mainLine.Color = blues(10,:);
fh.patch.FaceColor = blues(7,:);
% ylim([0.25,1])
xlim([0,400])
xlabel('Time','FontSize',16)
ylabel('PV correlation','FontSize',16)
set(gca,'FontSize',16,'LineWidth',1)

prefix = [figPre, 'pvCorrCoef'];
saveas(f_pvCorr,[sFolder,filesep,prefix,'.fig'])
print('-depsc',[sFolder,filesep,prefix,'.eps'])




% ************************************************************
% Representational similarity across time
% ************************************************************
% for better viusalization, only use part of the data
% gray color maps
% selInx = 1:10:1024;
selInx = find(~isnan(pks(:,1)));

% Y1= Yt(selInx,:,1);
% Y2 = Yt(selInx,:,1000);
Y1= Yt(:,:,1000);
Y2 = Yt(:,:,1300);


SM1 = Y1'*Y1;
SM2 = Y2'*Y2;
% SM1 = Y1(:,selInx)'*Y1(:,selInx);
% SM2 = Y2(:,selInx)'*Y2(:,selInx);

f_sm1 = figure;
set(f_sm1,'color','w','Units','inches')
pos(3)=3;  
pos(4)=3;
set(f_sm1,'Position',pos)

imagesc(SM1,[0,15]);
% colormap(GreyMap)
% cb = colorbar;
% set(cb,'FontSize',12)
set(gca,'XTick','','YTick','','LineWidth',0.5,'Visible','off')

prefix = [figPre, 'sm1'];
saveas(f_sm1,[sFolder,filesep,prefix,'.fig'])
print('-depsc',[sFolder,filesep,prefix,'.eps'])

f_sm2 = figure;
set(f_sm2,'color','w','Units','inches')
pos(3)=3;  
pos(4)=3;
set(f_sm2,'Position',pos)

imagesc(SM2,[0,15]);
% colormap(GreyMap)
% cb = colorbar;
% set(cb,'FontSize',12)
set(gca,'XTick','','YTick','','LineWidth',0.5,'Visible','off')
prefix = [figPre, 'sm2'];
saveas(f_sm2,[sFolder,filesep,prefix,'.fig'])
print('-depsc',[sFolder,filesep,prefix,'.eps'])


% heatmap of population vector odered by Y1
[pkSorted,sortedInx] = sort(Y1,2,'descend');
% significant nonzeor at time 1
actiTime1Inx = find(pkSorted(:,1)>0.05);

% [~,neurOrder] = sort(sortedInx(:,1));
[~,neurOrder] = sort(sortedInx(actiTime1Inx,1));



pv1Fig = figure;
pos(3)=3;  
pos(4)=3;
set(pv1Fig,'color','w','Units','inches','Position',pos)
% imagesc(Y1(neurOrder(250:end),:),[0,2])
imagesc(Y1(actiTime1Inx(neurOrder),:),[0,3])
set(gca,'Visible','off')
% colorbar
prefix = [figPre, 'pvHeatMap1'];
saveas(pv1Fig,[sFolder,filesep,prefix,'.fig'])
print('-depsc',[sFolder,filesep,prefix,'.eps'])


pv2Fig = figure;
pos(3)=3;  
pos(4)=3;
set(pv2Fig,'color','w','Units','inches','Position',pos)
imagesc(Y2(actiTime1Inx(neurOrder),:),[0,3])
% imagesc(Y2(neurOrder(250:end),:),[0,2])
set(gca,'Visible','off')

prefix = [figPre, 'pvHeatMap2_'];
saveas(pv2Fig,[sFolder,filesep,prefix,'.fig'])
print('-depsc',[sFolder,filesep,prefix,'.eps'])



% Example mean square displacement vs time
timeInterval = size(msds,1);
figure
plot((1:timeInterval)'*sep,msds(:,randperm(param.Np,3)))
xlabel('Time')
ylabel('$\langle \Delta r\rangle^2$','Interpreter','latex')

% histogram of effective diffusion constants
figure
histogram(Ds)
xlabel('$D_r$','Interpreter','latex')
ylabel('Count')
title('Diffusion constants')

% ********************************************************
% Diffusion constant versus average peak amplitude
% ********************************************************


%% Probability of shift of peak positions
psRange = param.ps;   % the range of ring positions
pks0 = pkCenterMass;
tps = [50,100,200,400,500,800];
quantiles = (1:3)/4*psRange;
probs = nan(length(tps),length(quantiles));
for i = 1:length(tps)
    diffLag = abs(pks0(:,tps(i)+1:end,:) - pks0(:,1:end-tps(i),:));
    for j = 1:length(quantiles)
        probs(i,j) = sum(diffLag(:)>quantiles(j))/length(diffLag(:));
    end
end


probShift = figure; 
set(probShift,'color','w','Units','inches')
pos(3)=3.8;  %17.4; %8.5;%pos(3)*1.5;
pos(4)=2.8;%pos(4)*1.5;
set(probShift,'Position',pos)
hold on
for i = 1:length(quantiles)
    plot(tps'*step,probs(:,i),'o-','MarkerSize',8,'Color',blues(1+3*i,:),...
        'MarkerFaceColor',blues(1+3*i,:),'MarkerEdgeColor',blues(1+3*i,:),'LineWidth',1.5)
end
hold off
box on

lg = legend('\Delta s >1/4 L','\Delta s>1/2 L','\Delta s >3/4 L');
set(lg,'FontSize',14)
xlabel('$\Delta t$','Interpreter','latex','FontSize',20)
ylabel('Probability','FontSize',20)
set(gca,'LineWidth',1,'FontSize',16)

% prefix = 'Tmaze_prob_shift_dist';
% saveas(gcf,[sFolder,filesep,prefix,'.fig'])
% print('-depsc',[saveFolder,filesep,prefix,'.eps'])

%% Statistics of shift of RF, Compare with experiment
% This part is intended to compare with ﻿Gonzalez et al Science 2019 paper
% based on unsorted data

psRange = param.ps;  %total length of the linearized track

% tps = [400,800,1500];
tps = [200,400,800];


quantiles = (1:3)/4*param.ps;
probs = nan(length(tps),length(quantiles));
edges = -200:10:200;
numBins = length(edges)-1; % this is the same as in the experiments

fractionSep = nan(length(tps),numBins);  % store the fraction of different shift
randSiftFrac = nan(length(tps),numBins);  % store randomized shift

shifDist = figure; 
set(shifDist,'color','w','Units','inches')
pos(3)=figWidth;  %17.4; %8.5;%pos(3)*1.5;
pos(4)=figHeight;%pos(4)*1.5;
set(shifDist,'Position',pos)
hold on
for i = 1:length(tps)
    temp = pkCenterMass(:,tps(i)+1:end,:) - pkCenterMass(:,1:end-tps(i),:);
    
    [f,xi] = ksdensity(temp(~isnan(temp))); 
%     histogram(temp(~isnan(temp)),'Normalization','pdf')
    plot(xi,f,'Color',blues(1+3*i,:),'LineWidth',2)
%     diffLag = min(pi, abs(temp(~isnan(temp))));
    diffLag = abs(temp(~isnan(temp)));
    
   [fractionSep(i,:),~]= histcounts(temp(~isnan(temp)),edges,'Normalization','probability');
    
    for j = 1:length(quantiles)
        probs(i,j) = sum(diffLag(:)>quantiles(j))/length(diffLag(:));
    end
end

% compared with random position
randPks = nan(size(pkCenterMass));
for i = 1:size(pkCenterMass,2)
    inx = find(~isnan(pkCenterMass(:,i)));
    randPks(inx,i) = psRange*rand(length(inx),1);
end
temp = pkCenterMass - randPks;
diffLagRnd = abs(temp(~isnan(temp)));
[f,xi] = ksdensity(temp(~isnan(temp))); 


% randomized shift
for i=1:length(tps)
    temp = cell(1,1);
    inx = find(~isnan(pkCenterMass(:,tps(i))));
    for j = 1:50
        rps = nan(size(pkCenterMass,1),1);
        rps(inx) = psRange*rand(length(inx),1);
        temp{j} = pkCenterMass(:,tps(i)) - rps;
    end
    rndSf = cat(1,temp{:});
    [randSiftFrac(i,:),~] = histcounts(rndSf(~isnan(rndSf)),edges,'Normalization','probability');
end

% [randSiftFrac,~] = histcounts(temp(~isnan(temp)),numBins,'Normalization','probability');
plot(xi,f,'Color',reds(8,:),'LineWidth',2)
hold off
box on
% xlim([-6.5,6.5])
lg = legend(['\Delta t=',num2str(tps(1))],['\Delta t=',num2str(tps(2))],...
    ['\Delta t=',num2str(tps(3))],'Random','Location','northeast');
set(lg,'FontSize',14)
xlabel('$\Delta s$','Interpreter','latex','FontSize',24)
ylabel('pdf','FontSize',24)
set(gca,'LineWidth',1,'FontSize',20)


%
% Histount of the probability
histFig = figure;
pos(3)=figWidth; 
pos(4)=figHeight;
set(histFig,'color','w','Units','inches','Position',pos)

hold on
for i = 1:length(tps)
    plot(fractionSep(i,:)','Color',blues(1+3*i,:),'LineWidth',plotLineWd)
    plot(randSiftFrac(i,:)','Color',reds(1+3*i,:),'LineWidth',plotLineWd)
end
box on
lg = legend(['\Delta t=',num2str(tps(1))],['\Delta t=',num2str(tps(2))],...
    ['\Delta t=',num2str(tps(3))],'Random');
set(lg,'FontSize',14)
% xlim([0,50])
xticks([1,(1:4)*numBins/4])
x = -1:0.5:1;
xticklabels({x});
xlabel('Centroid shift','FontSize',labelFont)
ylabel('Fraction','FontSize',labelFont)
set(gca,'FontSize',axisFont)

% save the figure
prefix = [figPre, 'shiftDistrFrac'];
saveas(histFig,[sFolder,filesep,prefix,'.fig'])
print('-depsc',[sFolder,filesep,prefix,'.eps'])



% Plot of cumulative shift
probShift = figure; 
set(probShift,'color','w','Units','inches')
pos(3)=3.8;  %17.4; %8.5;%pos(3)*1.5;
pos(4)=2.8;%pos(4)*1.5;
set(probShift,'Position',pos)
hold on
for i = 1:length(quantiles)
    plot(tps'*step,probs(:,i),'o-','MarkerSize',8,'Color',blues(1+3*i,:),...
        'MarkerFaceColor',blues(1+3*i,:),'MarkerEdgeColor',blues(1+3*i,:),'LineWidth',1.5)
end
hold off
box on

lg = legend('\Delta s >1/4 L','\Delta s>1/2 L','\Delta s >3/4 L');
set(lg,'FontSize',14)
xlabel('$\Delta t$','Interpreter','latex','FontSize',20)
ylabel('Probability','FontSize',20)
set(gca,'LineWidth',1,'FontSize',16)

%%
% probabilty of shift interval that are less than 10%
tps = 1:10:1500;
probs10 = nan(length(tps),1);
for i = 1:length(tps)
    temp = pkCenterMass(:,tps(i)+1:end,:) - pkCenterMass(:,1:end-tps(i),:);
    diffLag = abs(temp(~isnan(temp)));
    probs10(i) = sum(diffLag(:)<0.1*psRange)/length(diffLag(:));
end

% random
probRnd = sum(diffLagRnd(:)<0.1*psRange)/length(diffLagRnd(:));

% probability of > 10% of the length
frac10Fig = figure;
pos(3)=figWidth; 
pos(4)=figHeight;
set(frac10Fig,'color','w','Units','inches','Position',pos)

hold on
plot(tps',probs10,'LineWidth',3)
plot([0;tps(end)],[probRnd;probRnd],'k--','LineWidth',2)
hold off
box on
legend('Model','Random')
xlabel('Time','FontSize',labelFont)
ylabel({'Fraction shift','< 10% change'},'FontSize',labelFont)
set(gca,'LineWidth',axisLineWd,'FontSize',axisFont)

prefix = [figPre, 'shift10percent_'];
saveas(frac10Fig,[sFolder,filesep,prefix,'.fig'])
print('-depsc',[sFolder,filesep,prefix,'.eps'])

%% Neuron overlap
timeSeps = 5:5:1000;
neuronOL = nan(length(timeSeps),1);
for i = 1:length(length(timeSeps))
    neuronOL(i) = mean(pkFlags(:,timeSeps(i).*pkFlags(:,1)));
end


%%
figure
imagesc(pkAmp'*pkAmp)

% fraction of cells that are place cells
figure
plot((1:size(pkAmp,2))*step,mean(pkAmp > ampThd,1),'LineWidth',2)
xlabel('$t$','Interpreter','latex','FontSize',24)
ylabel('fraction of place cell','FontSize',24)
set(gca,'FontSize',20,'LineWidth',1.5,'YLim',[0,1])

% peak amplitude of single place cell
pcSel = randperm(param.Np,1); %randomly slect one place cell
figure
plot((1:size(pkAmp,2))*step,pkAmp(pcSel,:))
xlabel('$t$','Interpreter','latex')
ylabel('Peak amplitude')

% peak position evolve with time
tps = [1,200,400];  % select several time point to plot
figure
for i = 1:length(tps)
    pkMat = zeros(param.ps,param.ps);
    for j = 1:size(pks,1)
        if ~isnan(pks(j,tps(i)))
            pkMat(pks(j,tps(i))) = 1;
        end
    end
    subplot(1,length(tps),i)
    imagesc(pkMat)
    title(['iteration: ',num2str(step*tps(i))],'FontSize',20)
    xlabel('x position','FontSize',20)
    ylabel('y position','FontSize',20)
    set(gca,'FontSize',20,'LineWidth',1)
end

% calculate the overall peak position movement
xs = floor(pks/param.ps);
ys = mod(pks,param.ps);
dxs = xs - xs(:,1);
dys = ys - ys(:,1);
shiftEcul = sqrt(dxs.^2 + dys.^2)/param.ps;  % in unit of m
z = shiftEcul(~isnan(shiftEcul));
figure
histogram(z,30)
xlabel('peak shift $|\Delta r|$', 'Interpreter','latex','FontSize',24)
ylabel('count','FontSize',24)
set(gca,'LineWidth',1,'FontSize',20)
% change of the weight matrix, W
% figure
% for i = 1:size(allW,3)
%     subplot(1,size(allW,3),i)
%     imagesc(allW(:,:,i),[-0.02,0.1])
% end
% 
% % change of the weight matrix, M
% figure
% for i = 1:size(allM,3)
%     M = allM(:,:,i) - diag(diag(allM(:,:,i)));
%     subplot(1,size(allM,3),i)
%     imagesc(M,[-0.02,0.05])
% end
% 
% figure
% histogram(M(:))
% 
% 
% figure
% W = allW(:,:,1);
% histogram(W(:))

% changes of matrix norm
chgW = nan(size(allW,3)-1,1);
chgM = nan(size(allM,3)-1,1);
for i = 1:size(allW,3)-1
    chgW(i) = norm(allW(:,:,i+1) - allW(:,:,1))/norm(allW(:,:,1));
    chgM(i) = norm(allM(:,:,i+1) - allM(:,:,1))/norm(allM(:,:,1));
end


% relative change of W
figure
plot((1:size(allW,3)-1)'*timeGap,chgW,'LineWidth',2)
xlabel('iteration','FontSize',20)
ylabel('$$\frac{||W_t - W_0||_F^2}{||W_0||_F^2}$$','Interpreter','latex','FontSize',20)
set(gca,'FontSize',20,'LineWidth',1,'YLim',[0,0.4])

% relative change of M
figure
plot((1:size(allM,3)-1)'*timeGap,chgM,'LineWidth',2)
xlabel('iteration','FontSize',20)
ylabel('$$\frac{||M_t - M_0||_F^2}{||M_0||_F^2}$$','Interpreter','latex','FontSize',20)
set(gca,'FontSize',20,'LineWidth',1,'YLim',[0,0.7])

ws1 = squeeze(allW(10,33,:));
ws2 = squeeze(allW(39,87,:));

figure
plot(ws1)
hold on
plot(ws2)
hold off

figure
subplot(1,2,1)
imagesc(allW(:,:,1))
colorbar
subplot(1,2,2)
imagesc(allW(:,:,40))
colorbar

figure
imagesc(param.W)

% in term of individual palce cells
figure
plSel = randperm(param.Np,15);
pwhdl = tight_subplot(3,5);
for i = 1:15
    imagesc(pwhdl(i),reshape(param.W(plSel(i),:),24,25))
    pwhdl(i).XAxis.Visible = 'off';
    pwhdl(i).YAxis.Visible = 'off'; 
end


ms1 = squeeze(allM(10,33,:));
ms2 = squeeze(allM(39,87,:));

figure
plot(ms1)
hold on
plot(ms2)
hold off

figure
scatter(ms1,ms2)