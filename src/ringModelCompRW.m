function [nearestDist,numActi,centrPks] = ringModelCompRW(Np,sig,lr,alp, bs,iter)
% ring model drift of RFs compared with independent random walk


k = Np;   % number of neurons
n = 2;     % input dimensionality, 3 for Tmaze and 2 for ring

BatchSize = bs;

% default is a ring
learnType = 'snsm'; % snsm if using simple non-negative similarity matching

radius = 1;
t = 2000;  % total number of samples
sep = 2*pi/t;
X = [radius*cos(0:sep:2*pi);radius*sin(0:sep:2*pi)];


%% setup the learning parameters
% initialize the states

% Use offline learning to find the inital solution
params.W = 0.1*randn(k,n);
params.M = eye(k);      % lateral connection if using simple nsm
params.lbd1 = 0;        % regularization for the simple nsm, 1e-3
params.lbd2 = 0.05;     % default 1e-3

params.alpha = alp;     % should be smaller than 1 if for the ring model
params.beta = 1; 
params.gy = 0.05;       % update step for y
params.b = zeros(k,1);  % biase
params.learnRate = lr;  % learning rate for W and b
params.noise =  sig;    % stanard deivation of noise 

if strcmp(learnType, 'snsm')
    for i = 1:1e4
        y0 = 0.1*rand(k,BatchSize);
        inx = randperm(t,BatchSize);
        x = X(:,inx);           % randomly select one input
        [states, params] = MantHelper.nsmDynBatch(x,y0, params);
        y = states.Y;

        % update weight matrix
        params.W = (1-params.learnRate)*params.W + params.learnRate*y*x'/BatchSize...
            +sqrt(params.learnRate)*params.noise*randn(k,n);        
        params.M = max(0,(1-params.learnRate)*params.M + params.learnRate*y*y'/BatchSize + ...
            sqrt(params.learnRate)*params.noise*randn(k,k));
        params.b = (1-params.learnRate)*params.b + params.learnRate*sqrt(params.alpha)*mean(y,2);


    end
end


%% continue updating with noise
% BatchSize = 1;  % this used only during debug 
tot_iter = iter;
step = 10;
time_points = round(tot_iter/step);

% testing data, only used when check the representations
Xsel = X(:,1:10:end);        % modified on 7/18/2020
Y0 = zeros(k,size(Xsel,2));
% Yt = nan(k,size(Xsel,2),time_points);
ampThd = 0.05;       % threhold of place cell

pks = nan(k,time_points);    % store all the peak positions
pkAmp = nan(k,time_points);  % store the peak amplitude

if strcmp(learnType, 'snsm')
    for i = 1:tot_iter
        inx = randperm(t,BatchSize);
        y0 = 0.1*rand(k,BatchSize);
        x = X(:,inx);           % randomly select one input
        [states, params] = MantHelper.nsmDynBatch(x,y0,params);  % 1/20/2021
        y = states.Y;
        
        % update weight matrix   
        params.W = (1-params.learnRate)*params.W + params.learnRate*y*x'/BatchSize...
            +sqrt(params.learnRate)*params.noise*randn(k,n);        
        params.M = max(0,(1-params.learnRate)*params.M + params.learnRate*y*y'/BatchSize + ...
            sqrt(params.learnRate)*params.noise*randn(k,k));
        params.b = (1-params.learnRate)*params.b + params.learnRate*sqrt(params.alpha)*mean(y,2);
        
        if mod(i, step) == 0
            states_fixed = MantHelper.nsmDynBatch(Xsel,Y0,params);
            flag = sum(states_fixed.Y > ampThd,2) > 5;  % only those neurons that have multiple non-zeros
            [~,pkInx] = sort(states_fixed.Y,2, 'descend');
            temp = states_fixed.Y((pkInx(:,1)-1)*k + (1:k)');
            pkAmp(flag,round(i/step)) = temp(flag);
            pks(flag,round(i/step)) = pkInx(flag,1);
        end
    end 
end


% mean and variance of neighbor
nearestDist = nan(size(pks,2),2);  % store the mean and std
numActi = nan(size(pks,2),1);
L = size(Xsel,2);

% it only makes sense when multiple neurons are there
if Np > 1
    for i = 1:size(pks,2)
        temp = sort(pks(~isnan(pks(:,i)),i));
        ds = [diff(temp)/L*2*pi;(temp(1)-temp(end))/L*2*pi + 2*pi];
        nearestDist(i,:) = [mean(ds),var(ds)];
        numActi(i) = length(temp);
    end
end

% when N = 1, also return the peaks for comparison purpose
if nargout > 2
    centrPks = pks/L*2*pi;   % absolute position on the ring
end

end