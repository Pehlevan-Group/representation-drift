% This program demonstrates that learned representation can be maintained
% even without continuous access to sensory input if there are fast and
% slow forgetting timescale

close all
clear

%% setting for the graphics
% plot setting
defaultGraphicsSetttings

rb = brewermap(11,'RdBu');
set1 = brewermap(9,'Set1');

saveFolder = ['.',filesep,'figures'];

%% Generate sample data

params.Np = 50;           % number of neurons
params.dim_in = 2;              % input dimensionality, 3 for Tmaze and 2 for ring
total_iter = 1e5;               % total simulation iterations

% default is a ring
learnType = 'snsm';             % snsm if using simple non-negative similarity matching
params.batch_size = 1;          % default 1
params.record_step = 1000;
params.ampThd = 0.2;           % amplitude threshold to assgin a RF

% generate the ring data input, 2D
params.ps = 628;                % total number of angles
X = generate_ring_input(params.ps);

%% setup the learning parameters


Cx = X*X'/size(X,2);            % input covariance matrix

% store the perturbations
params.record_step = 100;       % only keep track of the information every 100 iteration

% initialize the states
y0 = zeros(params.Np,params.batch_size);
Wout = zeros(1,params.Np); % linear decoder weight vector


% estimate error
validBatch = 100;               % randomly select 100 to estimate the error
Y0val = zeros(params.Np,validBatch);

params.W = 0.5*randn(params.Np,params.dim_in);
params.M = eye(params.Np); % lateral connection if using simple nsm
params.Wslow = 0.1*randn(params.Np,params.dim_in);
params.Mslow = eye(params.Np);   

params.lbd1 = 0.0;              % regularization for the simple nsm, 1e-3
params.lbd2 = 0.02;             % default 1e-3

params.alpha = 0.3;             % should be smaller than 1 if for the ring model
params.beta = 1;                % 
params.gy = 0.05;               % update step for y
params.b = zeros(params.Np,1);      % bias
params.learnRate = 0.05;        % learning rate for W and b
params.learnSlow = 1e-4;        % much slower learning rate of weight
% params.read_lr = learnRate*10;     % learning rate of readout matrix

params.sigWmax = 0.005;    % the maxium noise level for the forward matrix
params.sigMmax = 0.005;    % maximum noise level of recurrent matrix
params.noiseW =  params.sigWmax*ones(params.Np,params.dim_in);   % stanard deivation of noise, same for all
params.noiseM =  params.sigMmax*ones(params.Np,params.Np);  

params.step  = 100;       % record every 100 iterations

[~, params] = ring_update_multi_timescale(X,total_iter,params);

% check the receptive field
Xsel = X(:,1:10:end);     % only use 10% of the data
Y0 = 0.1*rand(params.Np,size(Xsel,2));
Ys = nan(params.Np,size(Xsel,2));
for i = 1:size(Xsel,2)
%     states_fixed_nn = MantHelper.nsmDynBatchMultiScale(Xsel(:,i),Y0(:,i), params);
    states_fixed = PlaceCellhelper.nsmDynBatchMultiScale(Xsel(:,i),Y0(:,i), params);
    Ys(:,i) = states_fixed.Y;	
end

%%  Examine the decay of memory if without sensory input
% store the old weight matrices for future use
Wold = params.W; Mold = params.M;
total_iter =  1e5;  % iteration steps
[output, params] = ring_update_multi_timescale(X,total_iter,params,false);

% showing the tiling of RFs over time
time_sel = [1, 50, 100];
real_time_steps = [1, 5000, 1e4];
for i = 1:length(time_sel)
    Ys = output.Yt(:,:,time_sel(i));
    % order neurons based on their tuning position
%     [~,neuron_orde] = sort(Yt.pks(:,time_sel(i)),'ascend');
    [pkVal, peakPosi] = sort(Ys,2,'descend');
    [~,neuron_order] = sort(peakPosi(:,1),'ascend');
    
    % representations
    figure
    imagesc(Ys(neuron_order,:))
    colormap(viridis)
    colorbar
    title(['$t =', num2str(real_time_steps(i)),'$'],'Interpreter','latex')
    xlabel('$\theta$','Interpreter','latex','FontSize',24)
    ylabel('Sorted neuron','FontSize',24)
    set(gca,'FontSize',20,'XTick',[1,314,628],'XTickLabel',{'0','\pi','2\pi'})
    
    % similarity matrix
    figure
    imagesc(Ys'*Ys)
    colormap(viridis)
    colorbar
    title(['$t =', num2str(real_time_steps(i)),'$'],'Interpreter','latex')
    xlabel('$\theta$','Interpreter','latex','FontSize',24)
    ylabel('$\theta$','Interpreter','latex','FontSize',24)
    set(gca,'FontSize',20,'XTick',[1,314,628],'XTickLabel',{'0','\pi','2\pi'},...
        'YTick',[1,314,628],'YTickLabel',{'0','\pi','2\pi'})
    
    % multidimensional scaling
    Dist = pdist(Ys','euclidean');
    [Ymds,~] = cmdscale(Dist);
    figure
    plot(Ymds(:,1),Ymds(:,2),'.')
    xlabel('MDS 1')
    ylabel('MDS 2')
end


% now show the decay of memories if there is only fast synapses
params.W = Wold;  params.M = Mold;
params.dim_out = params.Np;   
[Yt_fast, params] = ring_update_weight(X,total_iter,params,false);

% plot the representations
for i = 1:length(time_sel)
    Ys = Yt_fast(:,:,time_sel(i));
    [~, peakPosi] = sort(Ys,2,'descend');
    [~,neuron_order] = sort(peakPosi(:,1),'ascend');
    
    
    % representation based on population vectors
    figure
    imagesc(Ys(neuron_order,:))
    colormap(viridis)
    colorbar
    xlabel('$\theta$','Interpreter','latex','FontSize',24)
    ylabel('Sorted neuron','FontSize',24)
    set(gca,'FontSize',20,'XTick',[1,314,628],'XTickLabel',{'0','\pi','2\pi'})
    
    % similarity matrix
    figure
    imagesc(Ys'*Ys)
    colormap(viridis)
    colorbar
    title(['$t =', num2str(real_time_steps(i)),'$'],'Interpreter','latex')
    xlabel('$\theta$','Interpreter','latex','FontSize',24)
    ylabel('$\theta$','Interpreter','latex','FontSize',24)
    set(gca,'FontSize',20,'XTick',[1,314,628],'XTickLabel',{'0','\pi','2\pi'},...
        'YTick',[1,314,628],'YTickLabel',{'0','\pi','2\pi'})
    
    % multidimensional scaling
    Dist = pdist(Ys','euclidean');
    [Ymds,~] = cmdscale(Dist);
    figure
    plot(Ymds(:,1),Ymds(:,2),'.')
    xlabel('MDS 1')
    ylabel('MDS 2')
end