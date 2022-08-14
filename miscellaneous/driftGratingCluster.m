function driftGratingCluster(noiseStd,lr, lbd1, lbd2, alp, totIter)
% representational drift in visual systems
% nonnegtive similarity mathching and drifting representation of drifting
% gratings related to Deitch, Rubin and Ziv bioRxiv 2020 paper "Representational 
% drift in the mouse visual cortex" 
% the input data is from a movie " touch of the evil", we only slect some
% of the first 30s frames
% this is a version for running on the cluster

% noiseStd      standard deviation of noise
% lbd1          l1 norm regularizer
% lbd2          l2 norm regularizer
% alp           threshold during similarity matching
% totIter       number of iteration during the noisy update phase


%% load the data
% when running on cluster
addpath('/n/home09/ssqin/representationDrift')

load('./data/gratings.mat');
newData = img;
X = reshape(newData,size(newData,1)*size(newData,2),size(newData,3));


%% setup the learning parameters

n = size(newData,1)*size(newData,2);   % number of neurons
dimOut = 200;            % 200 output neurons
t = size(newData,3);     % total number of frames
BatchSize = 1;

learnRate = lr;        % learning rate 0.05
params.gy = 0.05;         % update step for y

% initialize the states
y0 = zeros(dimOut,BatchSize);


% Use offline learning to find the inital solution
params.W = 0.01*randn(dimOut,n);
params.M = eye(dimOut);       % lateral connection if using simple nsm
params.lbd1 = lbd1;           % regularization for the simple nsm, 1e-5
params.lbd2 = lbd2;           %default 1e-3

params.alpha = alp;            % default 0.92
params.b = zeros(dimOut,1);    % bias
params.learnRate = learnRate;  % learning rate for W and b
params.noise =  noiseStd;      % stanard deivation of noise, default 1e-5


% initial phase, find the solution
for i = 1:1e3
    inx = mod(i,size(X,2));
    if inx ==0
        inx = size(X,2);
    end
    x = X(:,inx);
%     x = X(:,randperm(t,BatchSize));  % randomly select one input
    [states, params] = MantHelper.nsmDynBatch(x,y0, params);
    y = states.Y;

    % update weight matrix, only W with noise
    params.W = (1-params.learnRate)*params.W + params.learnRate*y*x'/BatchSize...
        +sqrt(params.learnRate)*params.noise*randn(dimOut,n);        
%     params.M = (1-params.learnRate)*params.M + params.learnRate*y*y'/BatchSize;
        params.M = max((1-params.learnRate)*params.M + params.learnRate*y*y'/BatchSize + ...
            sqrt(params.learnRate)*params.noise*randn(dimOut,dimOut),0);
    params.b = (1-params.learnRate)*params.b + params.learnRate*sqrt(params.alpha)*mean(y,2);
end

% check the receptive field
% Y0 = zeros(dimOut,size(X,2));
% [states_fixed_nn, params] = MantHelper.nsmDynBatch(X,Y0,params);



%% continue updating with noise

tot_iter = totIter;      % default 2e3
step = 5;
time_points = round(tot_iter/step);

% testing data, only used when check the representations
Xsel = X;        % modified on 10/21/2020
Y0 = zeros(dimOut,size(Xsel,2));


% store all the output gram at each time point
Yt = nan(dimOut,size(Xsel,2),time_points);

for i = 1:tot_iter
    x = X(:,randperm(t,BatchSize));  % randomly select one input
    [states, params] = MantHelper.nsmDynBatch(x,y0,params);
    y = states.Y;

    % update weight matrix
    params.W = (1-params.learnRate)*params.W + params.learnRate*y*x'/BatchSize + ...
        sqrt(params.learnRate)*params.noise*randn(dimOut,n);  % adding noise
    params.M = max((1-params.learnRate)*params.M + params.learnRate*y*y'/BatchSize + ...
            sqrt(params.learnRate)*params.noise*randn(dimOut,dimOut),0);
%     params.M = (1-params.learnRate)*params.M + params.learnRate*y*y'/BatchSize;
    params.b = (1-params.learnRate)*params.b + params.learnRate*sqrt(params.alpha)*mean(y,2);

    % store and check representations
    if mod(i, step) == 0
        [states_fixed, params] = MantHelper.nsmDynBatch(Xsel,Y0,params);
        Yt(:,:,round(i/step)) = states_fixed.Y;
    end
end 

% save the data

fileName = ['./data/driftGrating_ns', num2str(noiseStd),'_l1_',num2str(lbd1),'_l2_', num2str(lbd2),'.mat'];
save(fileName)


end