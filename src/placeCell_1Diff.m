function Ds = placeCell_1Diff(Np,noiseStd,lbd1,lbd2,alp,tot_iter,learnRate,rwStep, lnTy,varargin)
% using non-negative similarity matching to learn a 2-d place cells based
% on predefined grid cells patterns. Based on the paper lian et al 2020
% Model 1m x 1m 2d environemnt, number of modules of grid cells is 4

% this is a version running on cluster
% Np              an integer, number of place cells
% noiseStd        standard deviation of noise
% lbd1            l1 regularization
% lbd2            l2 regularization
% tot_iter        total number of iteration
% learnRate       learning rate
% rwStep          step size of random walk, default 1
% lnTy            learning type, a string, snsm or randwalk

if nargin > 9
    repId = varargin{1};
end

%% model parameters
param.ps =  200;      % number of positions along each dimension
param.Nlbd = 5;       % number of different scales/spacing
param.Nthe =  20;      % offset along the 1D
param.Ng = param.Nlbd*param.Nthe;   % total number of grid cells
param.Np = Np;   % number of place cells, default 20*20

param.baseLbd = 1/4;   % spacing of smallest grid RF, default 0.28
param.sf =  1.42;       % scaling factor between adjacent module

% parameters for learning 
% noiseStd = noiseStd;          % 0.05
% learnRate = learnRate;     % default 0.05

param.W = 0.5*randn(param.Np,param.Ng);   % initialize the forward matrix
param.M = eye(param.Np);        % lateral connection if using simple nsm
param.lbd1 = lbd1;               % 0.15 for 400 place cells and 5 modes
param.lbd2 = lbd2;              % 0.05 for 400 place cells and 5 modes


param.alpha = alp;  % 80 for regular,95 for 5 grid modes, 150 for weakly input
param.beta = 2; 
param.gy = 0.1;   % update step for y
param.gz = 0.1;   % update step for z
param.gv = 0.2;   % update step for V
param.b = zeros(param.Np,1);  % biase
param.learnRate = learnRate;  % learning rate for W and b
param.noise =  noiseStd;   % stanard deivation of noise 
param.rwSpeed = rwStep;         % steps each update, default 1
run_dir = 'right';    % runing right

BatchSize = 1;    % minibatch used to to learn
learnType = lnTy;  % snsm, batch, online, randwalk

gridQuality = 'regular';  % regular or weak


%% generate grid fields
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

%% using non-negative similarity matching to learng place fields
% generate input from grid filds
% all the position input by the grid code
posiGram = eye(param.ps);
gdInput = gridFields'*posiGram;  % gramin of input matrix

preparPhaseLen = 1e4;   % length of prepare phases

% simple non-negative similarity matching
if strcmp(learnType, 'snsm')    
    
    ystart = zeros(param.Np,BatchSize);  % inital of the output
    
    % generate position code by the grid cells    
    for i = 1:preparPhaseLen
        positions = gdInput(:,randperm(param.ps,BatchSize));
        states= PlaceCellhelper.nsmDynBatch(positions,ystart, param);
        y = states.Y;
        
        % update weight matrix
        param.M = (1-param.learnRate)*param.M + param.learnRate*y*y'/BatchSize;
        param.W = (1-param.learnRate)*param.W + param.learnRate*y*positions'/BatchSize + ...
            sqrt(param.learnRate)*param.noise*randn(param.Np,param.Ng);
        param.b = (1-param.learnRate)*param.b + param.learnRate*sqrt(param.alpha)*mean(y,2);

    end
 
% if input is drawn from a random walk
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
        
        states= PlaceCellhelper.nsmDynBatch(positions,ystart, param);
        y = states.Y;
        
        % update weight matrix
        param.M = (1-param.learnRate)*param.M + param.learnRate*y*y';
        param.W = (1-param.learnRate)*param.W + param.learnRate*y*positions' + ...
            sqrt(param.learnRate)*param.noise*randn(param.Np,param.Ng);
        param.b = (1-param.learnRate)*param.b + param.learnRate*sqrt(param.alpha)*mean(y,2);

    end
    
% if input is drawn from a random walk
elseif strcmp(learnType, 'direction')
    
    % position information is delivered as a random walk
    ystart = zeros(param.Np,1);  % inital of the output
    
    % generate position code by the grid cells
    posiInfo = nan(tot_iter,1);
    ix = randperm(param.ps,1);   % random select seed position
    if strcmp(run_dir, 'right')
        step_inter = 1;          % runing right
    elseif strcmp(run_dir, 'left')
        step_inter = -1;          % runign left
    end
    
    for i = 1:tot_iter
        ix = mod(ix + step_inter,param.ps);
        if ix ==0
            ix = param.ps;    % end point
        end
        posiInfo(i) = ix;
        positions = gdInput(:,ix);  % column-wise storation
        
        states= PlaceCellhelper.nsmDynBatch(positions,ystart, param);
        y = states.Y;
        
        % update weight matrix
        param.M = (1-param.learnRate)*param.M + param.learnRate*y*y';
        param.W = (1-param.learnRate)*param.W + param.learnRate*y*positions' + ...
            sqrt(param.learnRate)*param.noise*randn(param.Np,param.Ng);
        param.b = (1-param.learnRate)*param.b + param.learnRate*sqrt(param.alpha)*mean(y,2);

    end
end


%% Continuous Nosiy update
% tot_iter = 2e4;
% num_sel = 200;
step = 10;
time_points = round(tot_iter/step);

pks = nan(param.Np,time_points);    % store all the peak positions
pkAmp = nan(param.Np,time_points);  % store the peak amplitude
% placeFlag = nan(param.Np,time_points); % determine whether a place field
% store the weight matrices to see if they are stable

pkCenterMass = nan(param.Np,2,time_points);  % store the center of mass
pkMas = nan(param.Np,time_points);  

ampThd = 0.1;   % amplitude threshold, depending on the parameters


if strcmp(learnType, 'snsm')
    
    ystart = zeros(param.Np,BatchSize);  % inital of the output
    
    % generate position code by the grid cells    
    for i = 1:tot_iter
        positions = gdInput(:,randperm(param.ps,BatchSize));
        states = PlaceCellhelper.nsmDynBatch(positions,ystart, param);
        y = states.Y;
        
        % noisy update weight matrix
        param.W = (1-param.learnRate)*param.W + param.learnRate*y*positions'/BatchSize + ...
            sqrt(param.learnRate)*param.noise*randn(param.Np,param.Ng);  % 9/16/2020
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
%             Yt(:,:,round(i/step)) = states_fixed.Y;
            
            % store the center of mass
            [pkCM, aveMass] = PlaceCellhelper.centerMassPks1D(states_fixed.Y,ampThd);
            pkCenterMass(:,round(i/step)) = pkCM;
            pkMas(:,round(i/step)) = aveMass;

        end
    end
    
    % position information is delivered as a random walk
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
        Y0 = zeros(param.Np,size(gdInput,2));
        if mod(i, step) == 0
            states_fixed = PlaceCellhelper.nsmDynBatch(gdInput,Y0, param);
            flag = sum(states_fixed.Y > ampThd,2) > 3;  % only those neurons that have multiple non-zeros
            [~,pkInx] = sort(states_fixed.Y,2, 'descend');
            temp = states_fixed.Y((pkInx(:,1)-1)*param.Np + (1:param.Np)');
            pkAmp(flag,round(i/step)) = temp(flag);
            pks(flag,round(i/step)) = pkInx(flag,1);
            
            % store the center of mass
            [pkCM, aveMass] = PlaceCellhelper.centerMassPks1D(states_fixed.Y,ampThd);
            pkCenterMass(:,round(i/step)) = pkCM;
            pkMas(:,round(i/step)) = aveMass;

        end

    end

  % take input from "left" or "right" run
elseif strcmp(learnType, 'direction')
    
    ystart = zeros(param.Np,BatchSize);  % inital of the output
    
    
    % generate position code by the grid cells
    posiInfo = nan(tot_iter,1);
    ix = randperm(param.ps,1);   % random select seed position
    if strcmp(run_dir, 'right')
        step_inter = 1;          % runing right
    elseif strcmp(run_dir, 'left')
        step_inter = -1;          % runign left
    end
    
    % generate position code by the grid cells    
    for i = 1:tot_iter
        ix = mod(ix + step_inter,param.ps);
        if ix ==0
            ix = param.ps;    % end point
        end
        posiInfo(i) = ix;
        positions = gdInput(:,ix);  % column-wise storation
        
        states = PlaceCellhelper.nsmDynBatch(positions,ystart, param);
        y = states.Y;
        
        % noisy update weight matrix
        param.W = (1-param.learnRate)*param.W + param.learnRate*y*positions'/BatchSize + ...
            sqrt(param.learnRate)*param.noise*randn(param.Np,param.Ng);  % 9/16/2020
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
            [pkCM, aveMass] = PlaceCellhelper.centerMassPks1D(states_fixed.Y,ampThd);
            pkCenterMass(:,round(i/step)) = pkCM;
            pkMas(:,round(i/step)) = aveMass;

        end

    end
end

% Estimate the diffusion constants
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
Ds = PlaceCellhelper.fitLinearDiffusion(msds,step,100,'linear');


% save all the data
% sFile = ['./data/pcDiff1D_',learnType,'_Np_',num2str(Np),'_sig_',num2str(noiseStd),...
%     '_step',num2str(rwStep),'_lr',num2str(learnRate),'.mat'];
% save(sFile)
end