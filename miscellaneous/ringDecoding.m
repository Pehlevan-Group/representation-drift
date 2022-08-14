function errOut = ringDecoding(decodeRate, Np, alp, noiseStd, learnRate)
% This function systematically change the decoding learning rate to see how
% online readout performance change

k = Np;   % number of neurons
n = 2;     % input dimensionality, 3 for Tmaze and 2 for ring


BatchSize = 1;  % batch size

% default is a ring
learnType = 'snsm'; % snsm if using simple non-negative similarity matching

radius = 1;
t = 2000;  % total number of samples
sep = 2*pi/t;
X = [radius*cos(0:sep:2*pi);radius*sin(0:sep:2*pi)];


%% setup the learning parameters
% initialize the states
y0 = zeros(k,BatchSize);
Wout = zeros(1,k);      % linear decoder weight vector

% estimate error
validBatch = 100; % randomly select 100 to estimate the error
Y0val = zeros(k,validBatch);

% Use offline learning to find the inital solution
params.W = 0.1*randn(k,n);
params.M = eye(k);      % lateral connection if using simple nsm
params.lbd1 = 0;     % regularization for the simple nsm, 1e-3
params.lbd2 = 0.05;        %default 1e-3

params.alpha = alp;  % should be smaller than 1 if for the ring model
params.beta = 1; 
params.gy = 0.1;   % update step for y
params.b = zeros(k,1);  % biase
params.learnRate = learnRate;  % learning rate for W and b
params.noise =  noiseStd;   % stanard deivation of noise 

params.read_lr = decodeRate;     % learning rate of readout matrix

readError = [];   % store the readout error of a linear decoder

% modular of the output to 2\pi
phix = @(x) min(2*pi-abs(x),abs(x));


if strcmp(learnType, 'snsm')
    for i = 1:3e4
        inx = randperm(t,BatchSize);
        x = X(:,inx);  % randomly select one input
        [states, params] = MantHelper.nsmDynBatch(x,y0, params);
        y = states.Y;

        % update weight matrix
%         params.W = (1-params.learnRate)*params.W + params.learnRate*y*x'/BatchSize;
        params.W = (1-params.learnRate)*params.W + params.learnRate*y*x'/BatchSize...
            +sqrt(params.learnRate)*params.noise*randn(k,n);        
        params.M = (1-params.learnRate)*params.M + params.learnRate*y*y'/BatchSize;
%         params.M = (1-params.learnRate)*params.M + params.learnRate*y*y'/BatchSize + ...
%             sqrt(params.learnRate)*params.noise*randn(k,k);
        params.b = (1-params.learnRate)*params.b + params.learnRate*sqrt(params.alpha)*mean(y,2);

        % Assume a linear readout
        
        label = inx/t*2*pi - pi;  % from -pi to pi
%         label = inx/t*pi;  % from -pi to pi
%         dx = label - Wout*y;
%         Wout = Wout + params.read_lr*(sin(dx/2)).*cos(dx/2)/2*y'/BatchSize;
        Wout = Wout + params.read_lr*(label - Wout*y)*y'/BatchSize;
        
        if mod(i,100) == 1
            valSel = randperm(t,validBatch);
            valLabels = valSel/t*2*pi - pi;
            [states, ~] = MantHelper.nsmDynBatch(X(:,valSel),Y0val, params);
            readError = [readError, mean((valLabels - Wout*states.Y).^2)];
%             readError = [readError, mean(sin((valLabels - Wout*states.Y)/2).^2)];
        end
    end
end

% check the receptive field
% Xsel = X(:,1:2:end);
% Y0 = zeros(k,size(Xsel,2));
% [states_fixed_nn, ~] = MantHelper.nsmDynBatch(Xsel,Y0, params);



%============== Check the readout performance ===============
% valSel = randperm(t,validBatch);
% valLabels = valSel/t*2*pi - pi;
% [states, ~] = MantHelper.nsmDynBatch(X(:,valSel),Y0val, params);
% z = Wout*states.Y;
% 
% figure
% plot(valLabels,z,'o')
% xlabel('True position (rad)')
% ylabel('Predicted position (rad)')
% 
% 
% figure
% plot(readError)
% xlabel('Time')
% ylabel('Mean square error')

errOut = [mean(readError(end-100:end)), std(readError(end-100:end))];


end
