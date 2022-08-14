function Ds = placeCellClusterDiff(Np,noiseStd,lbd1,lbd2,alp,tot_iter,learnRate,rwStep, lnTy,varargin)
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

% learnRate = 0.01;     % default 0.05

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

BatchSize = 10;         % minibatch used to to learn
learnType = lnTy;    % snsm, batch, online, randwalk

% only used when using "bathc"
numIN  = 10;              % number of inhibitory neurons
Z0 = zeros(numIN,BatchSize);  % initialize interneurons
Y0 = zeros(param.Np, BatchSize); 
V = rand(numIN,param.Np);          % feedback from cortical neurons

gridQuality = 'regular';  % regular or weak

%% generate grid fields
if strcmp(gridQuality,'regular')
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

elseif strcmp(gridQuality,'weak')
    % generate weakly-tuned MEC cells
    sig = 2;
    gridFields = PlaceCellhelper.weakMEC(param.ps, param.Ng, sig);
    
%     % randomly select 15 of them to show
%     figure
%     ha = tight_subplot(3,5);
%     sel_inx = randperm(param.Ng, 15);
%     for i = 1:15
%         gd = gridFields(:,sel_inx(i));
%         imagesc(ha(i),reshape(gd,param.ps,param.ps))
%         ha(i).XAxis.Visible = 'off';
%         ha(i).YAxis.Visible = 'off';
%     end

end
%% using non-negative similarity matching to learng place fields
% generate input from grid filds
% all the position input by the grid code
posiGram = eye(param.ps*param.ps);
gdInput = gridFields'*posiGram;  % gramin of input matrix

preparPhaseLen = 1e4;   % length of prepare phases
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
    for i = 1:preparPhaseLen
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
    for i = 1:preparPhaseLen
        positions = gdInput(:,randperm(param.ps*param.ps,BatchSize));
        states= PlaceCellhelper.nsmDynBatch(positions,ystart, param);
        y = states.Y;
        
        % update weight matrix
%         param.W = (1-param.learnRate)*param.W + param.learnRate*y*positions'/BatchSize;
%         param.M = (1-param.learnRate)*param.M + param.learnRate*y*y'/BatchSize;
        param.W = (1-param.learnRate)*param.W + param.learnRate*y*positions'/BatchSize + ...
            sqrt(param.learnRate)*param.noise*randn(param.Np,param.Ng);
        param.M = max(0,(1-param.learnRate)*param.M + param.learnRate*y*y'/BatchSize + ...
            sqrt(param.learnRate)*param.noise*randn(param.Np,param.Np));
%         param.M = (1-param.learnRate)*param.M + param.learnRate*y*y'/BatchSize + ...
%             sqrt(param.learnRate)*param.noise*randn(param.Np,param.Np);

%         param.W = max((1-param.learnRate)*param.W + param.learnRate*y*gdInput'/BatchSize,0);
        param.b = (1-param.learnRate)*param.b + param.learnRate*sqrt(param.alpha)*mean(y,2);
    end
    
% if input is drawn from a random walk
elseif strcmp(learnType, 'randwalk')
    % position information is delivered as a random walk
    ystart = zeros(param.Np,1);  % inital of the output
    
    % generate position code by the grid cells
    ix = randperm(param.ps,1);
    iy = randperm(param.ps,1);
    
    posiInfo = nan(preparPhaseLen,2);
    for i = 1:preparPhaseLen
        [ix,iy] = PlaceCellhelper.nextPosi(ix,iy,param);
        posiInfo(i,:) = [ix,iy];
        positions = gdInput(:,(iy - 1)*param.ps + ix);  % column-wise storation
%         positions = gdInput(:,randperm(param.ps*param.ps,1));
        
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

% timeGap = 50;
% allW = nan(param.Np,param.Ng,round(tot_iter/timeGap));
% allM = nan(param.Np,param.Np,round(tot_iter/timeGap));
% allY = nan(param.Np,param.ps^2,round(tot_iter/timeGap)); % store all the population vectors

ampThd = 0.1;   % amplitude threshold, depending on the parameters

Y0 = zeros(param.Np,size(gdInput,2));
Z0 = zeros(numIN,size(gdInput,2));
Yt = nan(param.Np,param.ps*param.ps,time_points);
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
        positions = gdInput(:,randperm(param.ps*param.ps,BatchSize));
        states = PlaceCellhelper.nsmDynBatch(positions,ystart, param);
        y = states.Y;
        
        % noisy update weight matrix
        param.W = (1-param.learnRate)*param.W + param.learnRate*y*positions'/BatchSize + ...
            sqrt(param.learnRate)*param.noise*randn(param.Np,param.Ng);  % 9/16/2020
%         param.M = (1-param.learnRate)*param.M + param.learnRate*y*y'/BatchSize;
        param.M = max(0,(1-param.learnRate)*param.M + param.learnRate*y*y'/BatchSize + ...
            sqrt(param.learnRate)*param.noise*randn(param.Np,param.Np));
%         param.M = (1-param.learnRate)*param.M + param.learnRate*y*y'/BatchSize + ...
%             sqrt(param.learnRate)*param.noise*randn(param.Np,param.Np);
        param.b = (1-param.learnRate)*param.b + param.learnRate*sqrt(param.alpha)*mean(y,2);
%         param.b = (1-param.learnRate)*param.b + param.learnRate*sqrt(param.alpha)*mean(y,2) + ...
%             sqrt(param.learnRate)*param.noise*randn(param.Np,1);
        
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
    
    % position information is delivered as a random walk
elseif strcmp(learnType, 'randwalk')
    ystart = zeros(param.Np,1);  % inital of the output
    
    % generate position code by the grid cells
%     ix = randperm(param.ps,1);
%     iy = randperm(param.ps,1);
    
    posiInfo = nan(tot_iter,2);
    for i = 1:tot_iter
        [ix,iy] = PlaceCellhelper.nextPosi(ix,iy,param);
        posiInfo(i,:) = [ix,iy];
        positions = gdInput(:,(iy - 1)*param.ps + ix);  % column-wise storation
%         positions = gdInput(:,randperm(param.ps*param.ps,1));
        
        states= PlaceCellhelper.nsmDynBatch(positions,ystart, param);
        y = states.Y;
        
        % update weight matrix
        param.M = (1-param.learnRate)*param.M + param.learnRate*y*y';
        param.W = (1-param.learnRate)*param.W + param.learnRate*y*positions' + ...
            sqrt(param.learnRate)*param.noise*randn(param.Np,param.Ng);
%         param.M = (1-param.learnRate)*param.M + param.learnRate*y*y'/BatchSize + ...
%             sqrt(param.learnRate)*param.noise*randn(param.Np,param.Np);
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
% saturateMSD = msds(end,:);  % saturation mean square displacement


% save all the data
% sFile = ['./data/pcDiff_randwalk_Np_',num2str(Np),'_',...
%     num2str(repId),'.mat'];
sFile = ['./data/pc2D_Batch_Np_',num2str(Np),'std',num2str(noiseStd),'_l1_',num2str(lbd1),'_l2_',...
    num2str(lbd2),'_step',num2str(rwStep),'_lr',num2str(learnRate),'_',num2str(repId),'_',date,'.mat'];
save(sFile, '-v7.3')
end