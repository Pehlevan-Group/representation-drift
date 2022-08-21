function [dph,ephi] = rotaDiffConst(dimIn, dimOut, eigs, noiseStd, learnRate, varargin)
% this function returns the rotational diffusion constants calculated by two different
% methods    one based ont the mean squre displacement of the eucleadian
% distance   the other one based on the MSD of rotation angles
% dimIn      dimensionality of the input, an integer
% dimOut     dimensionality of the output, an integer
% eigs       used to generate correlated input, same length as the dimIn
% noiseStd   standard deviation of the noise added to weight updates
% learnRate  leraning rate
% dph        return the rotational diffusion costants
% ephi       return the exponents of rmsd

% by default, using the offline update of W and M, to emphasize the drift
% due to external noise
% last revised 08/20/2022

if nargin > 5
    tau = varargin{1};   % scaling factor of recurrent learning rate
else
    tau = 0.5;
end

if nargin > 6
    learnType = varargin{2};   % online or offline, default offline 
else
    learnType = 'online';      % default online learning
end

% basic parameters
n = dimIn;              % default 10
k = dimOut;             % default 3
t = 10000;              % total number of samples


% generate input data
V = orth(randn(n,n));
C = V*diag(eigs)*V';

% generate multivariate Gaussian distribution with specified correlation
% matrix
X = mvnrnd(zeros(1,n),C,t)';

% store the perturbations
% record_step = 200;
% perturbs = zeros(k,n,round(t/record_step));  %store the perturbation part of F


% Use offline learning to find the inital solution
dt = learnRate; 
W = randn(k,n);
M = eye(k,k);
noise1 = noiseStd;
noise2 = noiseStd;
dt0 = 0.1;         % learning rate for the initial phase
tau0 = 0.5;

% A transient offline learning to ensure starting point is stationary
for i = 1:500
    Y = pinv(M)*W*X; 
    W = (1-2*dt0)*W + 2*dt0*Y*X'/t;
    M = (1-dt0/tau0)*M + dt0/tau0*(Y*Y')/t;
end


% now add noise to see how representation drift
tot_iter = 5e4;     % total iteration time, longer for better estimation
num_sel = 500;      % only select part of the samples to estimate diffusion costant
step = 10;          % store every 10 step to reduce data size
time_points = round(tot_iter/step);
Yt = zeros(k,time_points,num_sel);
sel_inx = randperm(t,num_sel);

% use online or offline learning
if strcmp(learnType,'offline')
    for i = 1:tot_iter
        % sample noise matrices
        Y = pinv(M)*W*X; 
        W = (1-2*dt)*W + 2*dt*Y*X'/t + sqrt(2*dt)*noise1*randn(k,n);
        M = (1-dt/tau)*M + dt/tau*Y*Y'/t +  sqrt(dt/tau)*noise2*randn(k,k);

        if mod(i,step)==0
            Yt(:,round(i/step),:) = pinv(M)*W*X(:,sel_inx);
        end
    end
elseif strcmp(learnType,'online')
    for i = 1:tot_iter
        inx = randperm(t,1); % randomly select one input data
        Y = pinv(M)*W*X(:,inx); % 
        W = (1-2*dt)*W + 2*dt*Y*X(:,inx)' + sqrt(2*dt)*noise1*randn(k,n);
        M = (1-dt/tau)*M + dt/tau*Y*Y' +  sqrt(dt/tau)*noise2*randn(k,n); 
        
        % store every "step" step
        if mod(i,step)==0
           Yt(:,round(i/step),:) = pinv(M)*W*X(:,sel_inx);
        end       
    end
end

% estimate the rotational diffusion constants, notice the time unit is
% "step"
rmsd = SMhelper.rotationalMSD(Yt);

% return the diffusion constant
[dph,ephi] = SMhelper.fitRotationDiff(rmsd,step, 2000, 'log');

end