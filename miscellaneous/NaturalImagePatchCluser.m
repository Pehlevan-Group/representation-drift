function NaturalImagePatchCluser(totPatch,noiseStd,BatchSize,lbd1,lbd2,varargin)
% Prepare natural scence patches
% data is downloaded from http://www.rctn.org/bruno/sparsenet/, which was
% originally used in the paper Olshausen and Field in Nature, vol. 381, pp. 607-609.
% This is a version for running on cluster, modified from
% "naturalScenePatch.m"

% totPatch        integer, total number of patches
% noiseStd        double, noise level, default 0
% BatchSize       integer, bach size, deault 5
% lbd1            double,  the l1 regularization strength
% lbd2            double, the l2 regularization strength

%% configuration when running on cluster
addpath('/n/home09/ssqin/representationDrift')

% start the parallel pool with 12 workers
% parpool('local', str2num(getenv('SLURM_CPUS_PER_TASK')));


%% prepare 16*16 patches
dFile = './data/IMAGES.mat';
load(dFile)
Ldim = 16;              % 16*16 patches
dims = size(IMAGES);
extraWhitten = 0;      %make extra whittening

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
    [U,S,~] = svd(sigma);
    epsilon = 1e-5;
    imgPatch = U * diag(1./sqrt(diag(S) + epsilon)) * U' * X;
end

%% setup the learning parameters

n = size(imgPatch,1);   % number of neurons
k = 256;                % 256 output neurons
t = size(imgPatch,2);   % total number of frames
totIter = 3e5;          % number of total iteration

% noiseStd = 1e-3;      % 0.005 for ring, 1e-3 for Tmaze
learnRate = 1e-3;       % learning rate at the beginning
params.gy = 0.1;        % update step for y


% store the perturbations
record_step = 2e3;
allW = cell(round(totIter/record_step),1);  % store all the forward matrices
allM = cell(round(totIter/record_step),1);  % all the lateral matrices
allb = cell(round(totIter/record_step),1);  % all the bias

% initialize the states
y0 = zeros(k,BatchSize);

% Use offline learning to find the inital solution
params.W = randn(k,n)/Ldim;
params.M = eye(k);         % lateral connection if using simple nsm
params.lbd1 = lbd1;        % regularization for the simple nsm, 1e-3
params.lbd2 = lbd2;        % default 1e-3

params.alpha = 0;       % default 0.9
params.b = zeros(k,1);  % biase
params.learnRate = learnRate;  % learning rate for W and b
params.noise =  noiseStd;   % stanard deivation of noise 


for i = 1:totIter
    x = imgPatch(:,randperm(t,BatchSize));  % randomly select one input
    [states, params] = MantHelper.nsmDynBatch(x,y0, params);
    y = states.Y;
    
    % using decaying learning rate
%     params.learnRate = learnRate*(1/(1+i/1e3));
    if i <= 1e4
        params.learnRate = 1e-3;
    elseif i <= 1e5
        params.learnRate = 1e-5;
    else
        params.learnRate = 5e-6;
    end
    
        
    % update weight matrix
    params.W = (1-params.learnRate)*params.W + params.learnRate*y*x'/BatchSize...
        +sqrt(params.learnRate)*params.noise*randn(k,n);        
    params.M = (1-params.learnRate)*params.M + params.learnRate*y*y'/BatchSize...
        + sqrt(params.learnRate)*params.noise*randn(k,k);
    params.b = (1-params.learnRate)*params.b + params.learnRate*sqrt(params.alpha)*mean(y,2);
    
    if mod(i, record_step) == 0
        allW{round(i/record_step)} = params.W;
        allM{round(i/record_step)} = params.M;
        allb{round(i/record_step)} = params.b;
        
    end
end


Y0 = zeros(k,size(imgPatch,2));
[states_fixed_nn, params] = MantHelper.nsmDynBatch(imgPatch,Y0,params);

% save the data
fileName = ['./data/natScene_ns', num2str(noiseStd),'_l1_',num2str(lbd1),'_l2_', num2str(lbd2),'.mat'];
save(fileName)

end