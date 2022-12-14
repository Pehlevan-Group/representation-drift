% This program compares three different noise scenarios in the ring model
% full synaptic noise, only W or only M noise

close all
clear


%% Generate sample data

params.dim_out = 200;            % number of neurons
params.dim_in = 2;              % input dimensionality, 3 for Tmaze and 2 for ring
total_iter = 1e3;      % total simulation iterations

% default is a ring
dataType = 'ring';      % 'Tmaze' or 'ring';
learnType = 'snsm';     % snsm if using simple non-negative similarity matching
noiseVar = 'same';      % using different noise or the same noise level for each synpase
params.batch_size = 1;  % default 1

num_angle = 628;        % total number of angles
X = generate_ring_input(num_angle);

params.noiseStd = 0.002;      % 0.005 for ring, 1e-3 for Tmaze
params.learnRate = 0.005;     % default 0.05
params.record_step = 100;    % only keep track of the information every 100 iteration

% initialize the states
y0 = zeros(params.dim_out,params.batch_size);

% Use offline learning to find the inital solution
params.W = 0.1*randn(params.dim_out,params.dim_in);
params.M = eye(params.dim_out);          % lateral connection if using simple nsm
params.lbd1 = 0.0;          % regularization for the simple nsm, 1e-3
params.lbd2 = 0.05;         % default 1e-3

params.alpha = 0;           % should be smaller than 1 if for the ring model
params.beta = 1;            % 
params.gy = 0.05;           % update step for y
params.b = zeros(params.dim_out,1);      % bias

params.sigWmax = params.noiseStd;      % the maxium noise level for the forward matrix
params.sigMmax = params.noiseStd;      % maximum noise level of recurrent matrix

if strcmp(noiseVar, 'various')
    noiseVecW = 10.^(rand(params.dim_out,1)*2-2);
    params.noiseW = noiseVecW*ones(1,params.dim_in)*params.sigWmax;   % noise amplitude is the same for each posterior 
    noiseVecM = 10.^(rand(params.dim_out,1)*2-2);
    params.noiseM = noiseVecM*ones(1,params.dim_out)*params.sigMmax; 
else
    params.noiseW =  params.sigWmax*ones(params.dim_out,params.dim_in);   % stanard deivation of noise, same for all
    params.noiseM =  params.sigMmax*ones(params.dim_out,params.dim_out);   
end

[~, params] = ring_update_weight(X,total_iter,params);


% check the receptive field after the initial stage
Xsel = X(:,1:10:end);     % only use 10% of the data
Y0 = 0.1*rand(params.dim_out,size(Xsel,2));
Ys = nan(params.dim_out,size(Xsel,2));
for i = 1:size(Xsel,2)
    states_fixed_nn = MantHelper.nsmDynBatch(Xsel(:,i),Y0(:,i), params);
    Ys(:,i) = states_fixed_nn.Y;	
end

%% Continue simulation with three different noise scenarios

total_iter = 2e4;
all_Yts = cell(3,1);        % store all the ensemnbles
PV_corr_coefs = cell(3,1);  % store the population vector correlations 
param_struct = cell(3,1);
param_struct{1} = params;
param_struct{2} = params; param_struct{2}.noiseM = 0;
param_struct{3} = params; param_struct{3}.noiseW = 0;

% this is for parallel running
parfor simulation_type = 1:3
    switch simulation_type
        % full noise model
        case simulation_type
            disp(simulation_type)
            [output, ~] = ring_update_weight(X,total_iter,param_struct{simulation_type});
            all_Yts{simulation_type} = output;
            
    end
end

%% estimate the correlation coefficent of PVs

blues = brewermap(11,'Blues');
reds = brewermap(11,'Reds');
greys = brewermap(11,'Greys');
fig_colors = [blues([7,11],:);reds([7,11],:);greys([7,11],:)];

time_points = round(total_iter/params.record_step);
   
f_pvCorr = figure;
set(f_pvCorr,'color','w','Units','inches')
pos(3)=3.5;  
pos(4)=2.8;
set(f_pvCorr,'Position',pos)

for phase = 1:3
    pvCorr = zeros(size(all_Yts{phase},3),size(all_Yts{phase},2)); 
    for i = 1:size(all_Yts{phase},3)
        for j = 1:size(all_Yts{phase},2)
            temp = all_Yts{phase}(:,j,i);
            C = corrcoef(temp,all_Yts{phase}(:,j,1));
            pvCorr(i,j) = C(1,2);
        end
    end
    PV_corr_coefs{phase} = pvCorr;
    % plot
    fh = shadedErrorBar((1:size(pvCorr,1))'*params.record_step,pvCorr',{@mean,@std});
    box on
    set(fh.edge,'Visible','off')
    fh.mainLine.LineWidth = 3;
    fh.mainLine.Color = fig_colors(2*phase-1,:);
    fh.patch.FaceColor = fig_colors(2*phase,:);
end

% ylim([0.25,1])
xlim([0,100]*params.record_step)
xlabel('Time','FontSize',16)
ylabel('PV correlation','FontSize',16)
set(gca,'FontSize',16,'LineWidth',1)


% save the figure
% figPref = ['../figures',filesep,['ring_model_different_noise_N',num2str(params.dim_out), '_1_',date]];
% saveas(gcf,[figPref,'.fig'])
% print('-depsc',[figPref,'.eps'])

% save the data
% close all
% dataFile = ['../data',filesep,'ring_model_different_noise_1.mat'];
% save(dataFile)
