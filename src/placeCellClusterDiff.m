function Ds = placeCellClusterDiff(Np,noiseStd,lbd1,lbd2,alp,tot_iter,learnRate,rwStep,varargin)
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
else
    repId = 1;
end

%% model parameters
param.ps =  32;     % number of positions along each dimension
param.Nlbd = 5;     % number of different scales/spacing
param.Nthe = 6;     % number of rotations
param.Nx =  5;      % offset of x-direction
param.Ny = 5;       % offset of y-direction
param.Ng = param.Nlbd*param.Nthe*param.Nx*param.Ny;   % total number of grid cells
param.Np = Np;   % number of grid cells

param.baseLbd = 0.2; % spacing of smallest grid RF, default 0.28
param.sf =  1.42;    % scaling factor between adjacent module

param.W = 0.5*randn(param.Np,param.Ng);   % initialize the forward matrix
param.M = eye(param.Np);   % lateral connection if using simple nsm
param.lbd1 = lbd1;     % 0.15 for 400 place cells and 5 modes
param.lbd2 = lbd2;     % 0.05 for 400 place cells and 5 modes


param.alpha = alp;     % 80 for regular,95 for 5 grid modes, 150 for weakly input
param.beta = 2; 
param.gy = 0.05;        % update step for y
param.gz = 0.1;        % update step for z
param.gv = 0.2;        % update step for V
param.b = zeros(param.Np,1);  % biase
param.learnRate = learnRate;  % learning rate for W and b
param.noise =  noiseStd;   % stanard deivation of noise
param.rwSpeed = rwStep;     % random walk speed, each iteration, the agent walks 5 steps

BatchSize = 1;         % minibatch used to to learn

%% generate grid fields

% sample parameters for grid cells
param.lbds = param.baseLbd*(param.sf.^(0:param.Nlbd-1));   % all the spacings of different modules
param.thetas =(0:param.Nthe-1)*pi/3/param.Nthe;         % random sample rotations
param.x0  = (0:param.Nx-1)'/param.Nx*param.lbds;            % offset of x
param.y0  = (0:param.Ny-1)'/param.Ny*param.lbds;            % offset of

% generate a Gramian of the grid fields
gridFields = nan(param.ps^2,param.Ng);
count = 1;    % concantenate the grid cells
for i = 1:param.Nlbd
    for j = 1: param.Nthe
        for k = 1:param.Nx
            for l = 1:param.Ny
%                     r = [i/param.ps;j/param.ps];
                r0 = [param.x0(k,i);param.y0(l,i)];
                gridFields(:,count) = PlaceCellhelper.gridModule(param.lbds(i),...
                    param.thetas(j),r0,param.ps);
                count = count +1;
            end
        end
    end
end

%% using non-negative similarity matching to learng place fields
% generate input from grid filds
% all the position input by the grid code
posiGram = eye(param.ps*param.ps);
gdInput = gridFields'*posiGram;  % gramin of input matrix

preparPhaseLen = 1e4;   % length of prepare phases

ystart = zeros(param.Np,BatchSize);  % inital of the output

% generate position code by the grid cells    
for i = 1:preparPhaseLen
        positions = gdInput(:,randperm(param.ps*param.ps,BatchSize));
        states= PlaceCellhelper.nsmDynBatch(positions,ystart, param);
        y = states.Y;
        
        % update weight matrix
        param.W = (1-param.learnRate)*param.W + param.learnRate*y*positions'/BatchSize + ...
            sqrt(param.learnRate)*param.noise*randn(param.Np,param.Ng);
        param.M = max(0,(1-param.learnRate)*param.M + param.learnRate*y*y'/BatchSize + ...
            sqrt(param.learnRate)*param.noise*randn(param.Np,param.Np));
        param.b = (1-param.learnRate)*param.b + param.learnRate*sqrt(param.alpha)*mean(y,2);
end

%% Continuous Nosiy update
step = 10;
time_points = round(tot_iter/step);

pks = nan(param.Np,time_points);    % store all the peak positions
pkAmp = nan(param.Np,time_points);  % store the peak amplitude
pkCenterMass = nan(param.Np,2,time_points);  % store the center of mass
pkMas = nan(param.Np,time_points);  

ampThd = 0.1;   % amplitude threshold, depending on the parameters

Yt = nan(param.Np,param.ps*param.ps,time_points);

ystart = zeros(param.Np,BatchSize);  % inital of the output

% generate position code by the grid cells    
for i = 1:tot_iter
    positions = gdInput(:,randperm(param.ps*param.ps,BatchSize));
    states = PlaceCellhelper.nsmDynBatch(positions,ystart, param);
    y = states.Y;

    % noisy update weight matrix
    param.W = (1-param.learnRate)*param.W + param.learnRate*y*positions'/BatchSize + ...
        sqrt(param.learnRate)*param.noise*randn(param.Np,param.Ng);  % 9/16/2020
    param.M = max(0,(1-param.learnRate)*param.M + param.learnRate*y*y'/BatchSize + ...
        sqrt(param.learnRate)*param.noise*randn(param.Np,param.Np));
    param.b = (1-param.learnRate)*param.b + param.learnRate*sqrt(param.alpha)*mean(y,2);

    % store and check representations
    Y0 = zeros(param.Np,size(gdInput,2));
    if mod(i, step) == 0
        states_fixed = PlaceCellhelper.nsmDynBatch(gdInput,Y0, param);
        flag = sum(states_fixed.Y > ampThd,2) > 4;  % only those neurons that have multiple non-zeros
        [~,pkInx] = sort(states_fixed.Y,2, 'descend');
        temp = states_fixed.Y((pkInx(:,1)-1)*param.Np + (1:param.Np)');
        pkAmp(flag,round(i/step)) = temp(flag);
%             pcInx = pkAmp(:,round(i/step)) > ampThd & flag;  % find the place cells
        pks(flag,round(i/step)) = pkInx(flag,1);
        Yt(:,:,round(i/step)) = states_fixed.Y;

         % store the center of mass
        [pkCM, mass] = PlaceCellhelper.centerMassPks(states_fixed.Y,param, ampThd);
        pkCenterMass(:,:,round(i/step)) = pkCM;
        pkMas(:,round(i/step)) = mass;
    end

end

% Estimate the diffusion constants
msds = nan(floor(time_points/2),size(pkMas,1));
for i = 1:size(msds,1)
    
    xcoord = squeeze(pkCenterMass(:,1,:));
    ycoord = squeeze(pkCenterMass(:,2,:));
    if param.Np == 1
        t1 = abs(xcoord(i+1:end)' - xcoord(1:end-i)');
        t2 = abs(ycoord(i+1:end)' - ycoord(1:end-i)');
    else
        t1 = abs(xcoord(:,i+1:end,:) - xcoord(:,1:end-i,:));
        t2 = abs(ycoord(:,i+1:end,:) - ycoord(:,1:end-i,:));
    end
    
    dx = min(t1,param.ps - t1);
    dy = min(t2,param.ps - t2);
    msds(i,:) = nanmean(dx.^2 + dy.^2,2)';
end

% linear regression to get the diffusion constant of each neuron
% fitRange = 100;   % only select the first 400
Ds = PlaceCellhelper.fitLinearDiffusion(msds,step,'linear');

% save all the data
sFile = ['./data/pc2D_online_Np_',num2str(Np),'std',num2str(noiseStd),'_l1_',num2str(lbd1),'_l2_',...
    num2str(lbd2),'_step',num2str(rwStep),'_lr',num2str(learnRate),'_',num2str(repId),'_',date,'.mat'];
save(sFile, '-v7.3')
end