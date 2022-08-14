% This program test the performance of Hebbian homeostasis and recurrent
% connection in the decoding of a ring model
% This decoding mechanism is proposed by Rule and O'Leary bioRxiv 2021
% The drifing RFs is modeled as in "ringPlaceModel.m"

% last revised on 3/14/2021

close all
clear


%% Generate sample data and decoding neuron profile

k = 100;            % number of representation neurons
n = 2;              % input dimensionality, 3 for Tmaze and 2 for ring

nz = 50;          % number of readout neuron
zWid = 0.1;       % the relative width of readout RF, normalized to [0, 1]
totIter = 2e3;   % total iterations

% default is a ring
dataType = 'ring'; % 'Tmaze' or 'ring';
learnType = 'snsm'; % snsm if using simple non-negative similarity matching
decodeMethod = 'SFA';  % use Hebbian homesostasis decoding method, 'HebbianHomeo', or 'recurrent'
BatchSize = 20;    % default 50
angularVel = 1/200;   % angular velocity

radius = 1;
t = 500;  % total number of samples
sep = 2*pi/t;
ags =sep:sep:2*pi;
X = [radius*cos(ags);radius*sin(ags)];

% desired readout RF
y_centers = (0:(nz-1))*2*pi/nz;
Zhat = nan(nz,t);
thetas = (0:t-1)*sep;
ysig = 2*pi*zWid;
for i = 1:nz
    temp = min([abs(thetas - y_centers(i));2*pi-abs(thetas - y_centers(i))],[],1);
    Zhat(i,:) = exp(-temp.^2/2/ysig^2);
end

% the target variance
varZ0  = mean(var(Zhat,0,2));


figure
imagesc(Zhat)
xlabel('$\theta$','Interpreter','latex')
ylabel('Output')
set(gca,'LineWidth',1.5,'FontSize',20,'XTick',[1,250,500],'XTickLabel',{'0', '\pi', '2\pi'})

figure
plot(Zhat(5:10:50,:)')
xlabel('$\theta$','Interpreter','latex')
ylabel('Response')
set(gca,'LineWidth',1.5,'FontSize',20,'XTick',[1,250,500],'XTickLabel',{'0', '\pi', '2\pi'})


% plot input similarity matrix
SMX = X'*X;

figure
ax = axes;
imagesc(ax,SMX);
colorbar
ax.XTick = [1 5e3 1e4];
ax.YTick = [1 5e3 1e4];
ax.XTickLabel = {'0', '\pi', '\pi'};
ax.YTickLabel = {'0', '\pi', '\pi'};
xlabel('position','FontSize',28)
ylabel('position','FontSize',28)
set(gca,'FontSize',24)

figure
plot(X(1,:),X(2,:))

%% setup the learning parameters

noiseStd = 0.00;      % 0.005 for ring, 1e-3 for Tmaze
learnRate = 0.01;   % default 0.05
kappa = 50*learnRate;   % this is used in the recurrent decoding model

Cx = X*X'/t;  % input covariance matrix

% store the perturbations
record_step = 10;

% initialize the states
y0 = zeros(k,BatchSize);
% Wout = 0.01*rand(k,nz);      % linear decoder weight vector

% estimate error
% validBatch = 100; % randomly select 100 to estimate the error
Y0val = zeros(k,t);

% Use offline learning to find the inital solution
params.W = 0.5*randn(k,n);
params.M = eye(k);      % lateral connection if using simple nsm
params.lbd1 = 0.0;     % regularization for the simple nsm, 1e-3
params.lbd2 = 0.01;        %default 1e-3

params.alpha = 0;    % should be smaller than 1 if for the ring model
params.beta = 1; 
params.gy = 0.05;   % update step for y
params.b = zeros(k,1);  % biase
params.learnRate = learnRate;  % learning rate for W and b
params.noise =  noiseStd;   % stanard deivation of noise 
params.read_lr = 1e-2;     % learning rate of readout matrix
params.rho = 1e-2;         % regularization of readout recurrent matrix
params.kappa = 1e-2;        % regularization of the readout matrix
params.kp_r = 1e-4;         % regularization for the recurrent matrix
params.rho_d = 1e-3;        % used in nonlinear readout
readError = [];   % store the readout error of a linear decoder

% allW = []; %only used for debug

% modular of the output to 2\pi
phix = @(x) min(2*pi-abs(x),abs(x));


if strcmp(learnType, 'snsm')
    for i = 1:2e3
        y0 = 0.1*rand(k,BatchSize);
        inx = randperm(t,BatchSize);
        x = X(:,inx);  % randomly select one input
        [states, params] = MantHelper.nsmDynBatch(x,y0, params);
        y = states.Y;

        % update weight matrix
%         params.W = (1-params.learnRate)*params.W + params.learnRate*y*x'/BatchSize;
        params.W = (1-params.learnRate)*params.W + params.learnRate*y*x'/BatchSize...
            +sqrt(params.learnRate)*params.noise*randn(k,n);        
%         params.M = max((1-params.learnRate)*params.M + diag(params.learnRate*mean(y.*y,2) + ...
%             sqrt(params.learnRate)*params.noise*randn(k,1)),0);
        params.M = max((1-params.learnRate)*params.M + params.learnRate*y*y'/BatchSize + ...
            sqrt(params.learnRate)*params.noise*randn(k,k),0);
%         params.M = (1-params.learnRate)*params.M + params.learnRate*y*y'/BatchSize;
        params.b = (1-params.learnRate)*params.b + params.learnRate*sqrt(params.alpha)*mean(y,2);
               
%         if mod(i,100) == 1
            [states, ~] = MantHelper.nsmDynBatch(X,Y0val, params);
%             Yout = Wout'*states.Y;
% %             readError = [readError, mean(sin((valLabels - Wout*states.Y)/2).^2)];
%         end

    end
elseif strcmp(learnType, 'inputNoise')
    for i = 1:2e4
        y0 = 0.2*rand(k,BatchSize);
%         inx = randperm(t,BatchSize);
%         inx = max(mod(i,size(X,2)),1);
        x = [cos(angularVel*i);sin(angularVel*i)] + randn(2,BatchSize)*params.noise;  % randomly select one input and noise
        [states, params] = MantHelper.nsmDynBatch(x,y0, params);
        y = states.Y;

        % update weight matrix
        params.W = (1-params.learnRate)*params.W + params.learnRate*y*x'/BatchSize;
        params.M = (1-params.learnRate)*params.M + params.learnRate*y*y'/BatchSize;
        params.b = (1-params.learnRate)*params.b + params.learnRate*sqrt(params.alpha)*mean(y,2);
    end
end

% check the receptive field
% Xsel = X(:,1:10:end);
Y0 = 0.1*rand(k,size(X,2));
[states_fixed_nn, ~] = MantHelper.nsmDynBatch(X,Y0, params);

% Decode
if strcmp(decodeMethod,'linear')
    covY = states_fixed_nn.Y*states_fixed_nn.Y'/t;
    covYZ = states_fixed_nn.Y*Zhat'/t;
    covZ = Zhat*Zhat'/size(Zhat,2);

    Wout = pinv(covY + 0.01*eye(k))*covYZ;
    Zpred = Wout'*states_fixed_nn.Y;
elseif strcmp(decodeMethod,'nonlinear')
    Wout = ringRecurrOut(states_fixed_nn.Y,Zhat,params.kappa);
    R = ringRecurrOut(Zhat,Zhat,params.kp_r);
    Zpred = exp(Wout'*states_fixed_nn.Y);
elseif strcmp(decodeMethod,'SFA')
    covZ = Zhat*Zhat'/size(Zhat,2);
    covY = states_fixed_nn.Y*states_fixed_nn.Y'/t;
    
%     Wout = covZ*covZY*pinv(covY + 1e-3*eye(size(covY,1)));
%     Wout = ringRecurrOut(states_fixed_nn.Y,Zhat,params.kappa);
%     R = ringRecurrOut(Zhat,Zhat,params.kp_r);
%     Zpred = exp(Wout'*states_fixed_nn.Y);
%     Zpred = pinv(covZ)*Wout*states_fixed_nn.Y;
    maxIter = 1e4;
    errTol = 1e-4;
    err = inf;
    Wout = 0.1*randn(nz,k);
    Mout = eye(nz);
    eta = 0.05;
    count = 1;
    while count < maxIter
        Z = pinv(Mout)*Wout*states_fixed_nn.Y;
        covZY = Z*states_fixed_nn.Y'/t;
        dW =  2*eta*(covZY - Wout*covY);
        err = norm(dW)/norm(Wout)/eta;
        Wout = Wout + dW;
        Mout = Mout + eta/2*(Z*Z'/t - Mout);
        count = count + 1;
    end
end


% error = norm(Zhat -Zpred)/t;  % relative decoding error
% disp(['decoding error:',num2str(error)])


%% Continuous with noise and evaluate the decoding peformance

% desired readout

tot_iter = 1e4;
num_sel = 200;
step = 10;
time_points = round(tot_iter/step);

ystore = nan(tot_iter,1);
% store the weight matrices to see if they are stable
allW = nan(k,n,time_points);
allM = nan(k,k,time_points);
allbias = nan(k,time_points);

decodeError = nan(time_points,1);  % store the decoding errors

% testing data, only used when check the representations
Xsel = X;   % modified on 2/23/2021 
Y0 = zeros(k,size(Xsel,2));
Yt = nan(k,size(Xsel,2),time_points);
ampThd = 0.05;       % threhold of place cell, 2/23/2021

pks = nan(k,time_points);    % store all the peak positions
pkAmp = nan(k,time_points);  % store the peak amplitude

y_oneDay = nan(k,step*BatchSize);
count = 1;
if strcmp(learnType, 'snsm')
    for i = 1:tot_iter
        y0 = 0.1*rand(k,BatchSize);
        inx = randperm(t,BatchSize);
        x = X(:,inx);  % randomly select one input
        [states, params] = MantHelper.nsmDynBatch(x,y0,params);  % 1/20/2021
        y = states.Y;
        y_oneDay(:,(count-1)*BatchSize+1:count*BatchSize) = y;
%         ystore(i) = y;
        
        % update weight matrix   
        params.W = (1-params.learnRate)*params.W + params.learnRate*y*x'/BatchSize...
            +sqrt(params.learnRate)*params.noise*randn(k,n);        
% %         params.M = (1-params.learnRate)*params.M + diag(params.learnRate*mean(y.*y,2));
%         params.M = max((1-params.learnRate)*params.M + diag(params.learnRate*mean(y.*y,2) + ...
%             sqrt(params.learnRate)*params.noise*randn(k,1)),0);
        params.M = max((1-params.learnRate)*params.M + params.learnRate*y*y'/BatchSize + ...
            sqrt(params.learnRate)*params.noise*randn(k,k),0);
%         params.M = (1-params.learnRate)*params.M + params.learnRate*y*y'/BatchSize;
        params.b = (1-params.learnRate)*params.b + params.learnRate*sqrt(params.alpha)*mean(y,2);
        
        
        if strcmp(decodeMethod,'SFA') && i > 1
            z_curr =  pinv(Mout)*Wout*y;
            y_bar = y + y_old;
            z_bar = z_curr + z_old;
            Wout = Wout + 2*eta*(z_bar*y_bar' - Wout*y_bar*y_bar');
            Mout = Mout + eta/2*(z_bar*z_bar' - Mout);
        end
        y_old = y;
        z_old = pinv(Mout)*Wout*y;
        
        count = count + 1;

        if mod(i, step) == 0
            states_fixed = MantHelper.nsmDynBatch(Xsel,Y0,params);
            % recurrent update rule
%             covZ = Zpred*Zpred'/size(Zpred,2);
%             Zpred = Wout'*y_oneDay;
%             R = pinv(covZ + kappa*eye(nz))*covZ;
%             Zr = R'*Zpred;
%             Wout = Wout + params.read_lr*(y_oneDay*Zr'/step/BatchSize - y_oneDay*y_oneDay'/step/BatchSize*Wout);
            if strcmp(decodeMethod,'nonlinear')
                Zpred = exp(Wout'*y_oneDay);
                Zr = R'*Zpred;
                Wout = Wout + 0.05*y_oneDay*(Zr-Zpred)'/step/BatchSize - params.rho_d*Wout;
%                 Wout = Wout + 0.1*(y_oneDay*Zr'/step/BatchSize - Wout)+ params.rho_d*y_oneDay*(Zr-Zpred)'/step/BatchSize;
                
                Zpred = exp(Wout'*states_fixed.Y);
%                 states_fixed = MantHelper.nsmDynBatch(Xsel,Y0,params);
            % update the readout weight, only forward 
%             err_var = (varZ0 - mean(var(Zpred,0,2)))/varZ0;
%             Wout = Wout + err_var*(2*y_oneDay*y_oneDay'/step/BatchSize - eye(k))*Wout;
%             Wout = Wout + params.read_lr*(2*y_oneDay*y_oneDay'/step/BatchSize - eye(k))*Wout;
%             Wout = Wout + params.read_lr*(y_oneDay*Zr'/step/BatchSize - Wout) + ...
%                 params.rho*(y_oneDay*(Zr - Zpred)'/step/BatchSize);
            elseif strcmp(decodeMethod,'linear')
                Zpred = Wout'*y_oneDay;
                covZ = Zpred*Zpred'/size(Zpred,2);
%                 Zpred = Wout'*y_oneDay;
                R = pinv(covZ + kappa*eye(nz))*covZ;
                Zr = R'*Zpred;
                Wout = Wout + params.read_lr*(y_oneDay*Zr'/step/BatchSize - y_oneDay*y_oneDay'/step/BatchSize*Wout);
%                 states_fixed = MantHelper.nsmDynBatch(Xsel,Y0,params);
                Zpred = Wout'*states_fixed.Y;
            end
%             decodeError(round(i/step)) = norm(Zhat -Zpred)/size(Xsel,2);
            count = 1;
            y_oneDay = nan(k,step*BatchSize);
            
            flag = sum(states_fixed.Y > ampThd,2) > 5;  % only those neurons that have multiple non-zeros
            [~,pkInx] = sort(states_fixed.Y,2, 'descend');
            temp = states_fixed.Y((pkInx(:,1)-1)*k + (1:k)');
            pkAmp(flag,round(i/step)) = temp(flag);
            pks(flag,round(i/step)) = pkInx(flag,1);
            Yt(:,:,round(i/step)) = states_fixed.Y;
            
        end
    end 
elseif strcmp(learnType, 'inputNoise')
    for i = 1:tot_iter
        y0 = 0.1*rand(k,BatchSize);
        x = [cos(angularVel*i);sin(angularVel*i)] + randn(2,BatchSize)*params.noise;  % randomly select one input and noise
        [states, params] = MantHelper.nsmDynBatch(x,y0, params);
        y = states.Y;

        % update weight matrix
        params.W = (1-params.learnRate)*params.W + params.learnRate*y*x'/BatchSize;
        params.M = (1-params.learnRate)*params.M + params.learnRate*y*y'/BatchSize;
        params.b = (1-params.learnRate)*params.b + params.learnRate*sqrt(params.alpha)*mean(y,2);   
        
%         if mod(i, step) == 0
%             states_fixed = MantHelper.nsmDynBatch(Xsel,Y0,params);
%             flag = sum(states_fixed.Y > ampThd,2) > 5;  % only those neurons that have multiple non-zeros
%             [~,pkInx] = sort(states_fixed.Y,2, 'descend');
%             temp = states_fixed.Y((pkInx(:,1)-1)*k + (1:k)');
%             pkAmp(flag,round(i/step)) = temp(flag);
%             pks(flag,round(i/step)) = pkInx(flag,1);
%             Yt(:,:,round(i/step)) = states_fixed.Y;
%             
%         end
    end
end

figure
plot((1:length(decodeError))*step,decodeError/norm(Zhat)*size(Xsel,2),'LineWidth',3)
xlabel('Time')
ylabel('$\frac{||\hat{Z} - Z||_F}{||\hat{Z}||_F}$','Interpreter','latex')
set(gca,'XScale','log')
% Y0 = 0.1*rand(k,size(x,2));
% [states_fixed_nn, ~] = MantHelper.nsmDynBatch(X,Y0, params);
% 
% 
figure
imagesc(Zpred)
xlabel('$\theta$','Interpreter','latex')
ylabel('Output')
set(gca,'LineWidth',1.5,'FontSize',20,'XTick',[1,250,500],'XTickLabel',{'0', '\pi', '2\pi'})

% % decoding error
% Zpred = Wout'*states_fixed_nn.Y;
% error = norm(Zhat -Zpred)/t;  % relative decoding error
% disp(['decoding error:',num2str(error)])