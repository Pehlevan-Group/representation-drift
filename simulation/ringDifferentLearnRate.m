% Explore how different learning rates affect the diffusion constant of
% each RF
% We assign different learning rate for individual neurons
% This simulation generate data for Fig. 4D

close all
clear
%% Generate sample data

k = 100;   % number of neurons
n = 2;     % input dimensionality, 3 for Tmaze and 2 for ring

BatchSize = 1;    % default 50

% randomly assign learning rates
lrs = 5*10.^(2*rand(k,1)-4);  % randomly sampled from 0.001 to 0.1

% default is a ring
learnType = 'snsm'; % snsm if using simple non-negative similarity matching

radius = 1;
t = 1000;  % total number of samples
sep = 2*pi/t;
X = [radius*cos(0:sep:2*pi);radius*sin(0:sep:2*pi)];


%% setup the learning parameters
% initialize the states
y0 = zeros(k,BatchSize);

% Use offline learning to find the inital solution
params.W = 0.1*randn(k,n);
params.M = eye(k);      % lateral connection if using simple nsm
params.lbd1 = 0;     % regularization for the simple nsm, 1e-3
params.lbd2 = 0.05;        %default 1e-3

params.alpha = 0;         % should be smaller than 1 if for the ring model
params.beta = 1; 
params.gy = 0.05;         % update step for y
params.b = zeros(k,1);    % bias

params.lrW = lrs*ones(1,n);
params.lrM = lrs*ones(1,k);
params.noise =  0.01;      % stanard deivation of noise 


if strcmp(learnType, 'snsm')
    for i = 1:5e3
        inx = randperm(t,BatchSize);
        x = X(:,inx);  % randomly select one input
        states= MantHelper.nsmDynBatch(x,y0, params);
        y = states.Y;

        % update weight matrix
        params.W = (1-params.lrW).*params.W + params.lrW.*(y*x'/BatchSize)...
            +sqrt(params.lrW).*randn(k,n)*params.noise;        
%         params.M = (1-params.learnRate)*params.M + params.learnRate*y*y'/BatchSize;
        params.M = max(0,(1-params.lrM).*params.M + params.lrM.*(y*y'/BatchSize) + ...
            sqrt(params.lrM).*randn(k,k)*params.noise);
        params.b = (1-lrs).*params.b + lrs.*mean(y,2)*sqrt(params.alpha);


    end
end



%% continue updating with noise
% BatchSize = 1;  % this used only during debug 
tot_iter = 2e4;
step = 10;
time_points = round(tot_iter/step);

% store the weight matrices to see if they are stable
% allW = nan(k,n,time_points);

% testing data, only used when check the representations
Xsel = X(:,1:5:end);        % modified on 7/18/2020
Y0 = zeros(k,size(Xsel,2));
Yt = nan(k,size(Xsel,2),time_points);
ampThd = 0.01;       % threhold of place cell

pks = nan(k,time_points);    % store all the peak positions
pkAmp = nan(k,time_points);  % store the peak amplitude


% only for debug
% allWs = nan(Np,2,time_points);

if strcmp(learnType, 'snsm')
    for i = 1:tot_iter
        inx = randperm(t,BatchSize);
        x = X(:,inx);  % randomly select one input
        states = MantHelper.nsmDynBatch(x,y0,params);  % 1/20/2021
        y = states.Y;
%         ystore(i) = y;
        
        % update weight matrix   
        params.W = (1-params.lrW).*params.W + params.lrW.*(y*x'/BatchSize)...
            +sqrt(params.lrW).*randn(k,n)*params.noise;        
%         params.M = (1-params.learnRate)*params.M + params.learnRate*y*y'/BatchSize;
        params.M = max(0,(1-params.lrM).*params.M + params.lrM.*(y*y'/BatchSize) + ...
            sqrt(params.lrM).*randn(k,k)*params.noise);
        params.b = (1-lrs).*params.b + lrs.*mean(y,2)*sqrt(params.alpha);
        
        if mod(i, step) == 0
            states_fixed = MantHelper.nsmDynBatch(Xsel,Y0,params);
            flag = sum(states_fixed.Y > ampThd,2) > 5;  % only those neurons that have multiple non-zeros
            [~,pkInx] = sort(states_fixed.Y,2, 'descend');
            temp = states_fixed.Y((pkInx(:,1)-1)*k + (1:k)');
            pkAmp(flag,round(i/step)) = temp(flag);
            pks(flag,round(i/step)) = pkInx(flag,1);
%             allWs(:,:,round(i/step)) = params.W;
        end
    end 
end

% estimate the width of RF
RFwidth = mean(states_fixed.Y > ampThd,2);
% Estimate the diffusion constants

newPeaks = pks/size(Xsel,2)*2*pi;   % 11/21/2020

msds = nan(floor(time_points/2),k);
for i = 1:floor(time_points/2)
%     diffLag = newPeaks(:,i+1:end,:) - newPeaks(:,1:end-i,:);
    diffLag = min(abs(newPeaks(:,i+1:end,:) - newPeaks(:,1:end-i,:)),...
        2*pi - abs(newPeaks(:,i+1:end,:) - newPeaks(:,1:end-i,:)) );
    msds(i,:) = nanmean(diffLag.*diffLag,2);
end

% Ds = PlaceCellhelper.fitLinearDiffusion(msds,step,100,'linear');
Ds =  MantHelper.fitRingDiffusion(msds,step,'log');

% maximum msds
maxMSD = max(msds,[],1);


%% plot and save figures
greys = brewermap(11,'Greys');
blues = brewermap(11,'Blues');

sFolder = './figures';
pos = [0, 0, 4, 3];


% Ds vs learning rate
f_D_lr= figure;
set(f_D_lr,'color','w','Units','inches','Position',pos)

plot(lrs,Ds,'o','MarkerSize',6,'MarkerEdgeColor',greys(9,:),'Color',greys(9,:),...
    'LineWidth',2)
xlabel('$\eta$','Interpreter','latex','FontSize',20)
ylabel('$D$','Interpreter','latex','FontSize',20)
xlim([5e-4,1e-1])
set(gca,'FontSize',16,'LineWidth',1,'XScale','log','YScale','log','XTick',10.^(-4:-1))


prefix = ['ring_different_learnRate_sig',num2str(params.noise),'_3'];
saveas(f_D_lr,[sFolder,filesep,prefix,'.fig'])
print('-depsc',[sFolder,filesep,prefix,'.eps'])

% save the data
dFile = './data/revision/ring_diff_learnRate_0608.mat';
save(dFile)
%% correlation between peak shift
%{
% dphi = nan(size(newPeaks,1)-1,k);  % store the one time peak shift
temp = diff(newPeaks');
dphi = temp;
inx = abs(temp) > pi;
dphi(inx) = mod(temp(inx),-sign(temp(inx))*2*pi);
corr_dphi = [];
for i = 1:k-1
    for j=i+1:k
        id = ~isnan( dphi(:,i)) & ~isnan( dphi(:,j));
        if sum(id)>1
            temp = corrcoef(dphi(id,i),dphi(id,j));
            try
                corr_dphi = [corr_dphi,temp(1,2)];
            catch
                disp('error: dimension of tmep is not 2 by 2');
                disp(temp);
            end
        end
    end
end
%}