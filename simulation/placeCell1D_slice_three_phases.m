% using non-negative similarity matching to learn a 1-d place cells based
% on predefined grid cells patterns. Based on the paper lian et al 2020
% 1D grid fields are slices of 2D lattice of grid fields

% add input noise simulation
% last revised  6/30/2021
clear
close all

%% model parameters
param.ps =  200;        % number of positions along each dimension
param.Nlbd = 5;         % number of different scales/spacing
param.Nthe = 6;         % number of rotations
param.Nx =  4;          % offset of x-direction
param.Ny = 4;           % offset of y-direction
param.Ng = param.Nlbd*param.Nthe*param.Nx*param.Ny;   % total number of grid cells
param.Np = 200;         % number of place cells, default 20*20

param.baseLbd = 1/4;    % spacing of smallest grid RF, default 1/4
param.sf =  1.42;       % scaling factor between adjacent module

% parameters for learning 
noiseVar = 'same';      % using different noise or the same noise level for each synpase
noiseStd = 0.05;        % 0.001
learnRate = 0.02;       % default 0.05

param.W = 0.5*randn(param.Np,param.Ng);   % initialize the forward matrix
param.M = eye(param.Np);        % lateral connection if using simple nsm
param.lbd1 = 0.04;              % 0.15 for 400 place cells and 5 modes
param.lbd2 = 0.1;               % 0.05 for 400 place cells and 5 modes


param.alpha = 65;        % the threshold depends on the input dimension, 65
param.beta = 2; 
param.gy = 0.05;         % update step for y
param.gz = 0.1;          % update step for z
param.gv = 0.2;          % update step for V
param.b = zeros(param.Np,1);  % biase
param.learnRate = learnRate;  % learning rate for W and b
param.noise =  noiseStd; % stanard deivation of noise 
param.rwSpeed = 10;      % steps each update, default 1
param.step = 20;         % store every 20 step
param.ampThd = 0.1;      % amplitude threshold, depending on the parameters


param.sigWmax = noiseStd;% the maximum noise std of forward synapses
param.sigMmax = noiseStd;     
if strcmp(noiseVar, 'various')
%     noiseVecW = rand(param.Np,1);
    noiseVecW = 10.^(rand(param.Np,1)*2-2);
    param.noiseW = noiseVecW*ones(1,param.Ng)*param.sigWmax;   % noise amplitude is the same for each posterior 
%     noiseVecM = rand(param.Np,1);
    noiseVecM = 10.^(rand(param.Np,1)*2-2);
    param.noiseM = noiseVecM*ones(1,param.Np)*param.sigMmax; 
else
    param.noiseW =  param.sigWmax*ones(param.Np,param.Ng);    % stanard deivation of noise, same for all
    param.noiseM =  param.sigMmax*ones(param.Np,param.Np);   
end

param.ori = 1/10*pi;     % slicing orientation

param.BatchSize = 1;      % minibatch used to to learn
param.learnType = 'snsm';  % snsm, batch, online, randwalk, direction, inputNoise

gridQuality = 'slice';  % regular, weak or slice

makeAnimation = 0;    % whether make a animation or not

gridFields = slice2Dgrid(param.ps,param.Nlbd,param.Nthe,param.Nx,param.Ny,param.ori);


%% using non-negative similarity matching to learn place fields
% generate input from grid filds

total_iter = 5e3;   % total interation, default 2e3

posiGram = eye(param.ps);
gdInput = gridFields'*posiGram;  % gramin of input matrix

% Fir the initial stage, only return the updated parameters
[~, param] = place_cell_stochastic_update_snsm(gdInput,total_iter, param);

%% Continuous Nosiy update
% we will have three different scenarios
% full noise, either W or M
total_iter = 1e4;
all_Yts = cell(3,1);        % store all the ensemnbles
PV_corr_coefs = cell(3,1);  % store the population vector correlations 
param_struct = cell(3,1);
param_struct{1} = param;
param_struct{2} = param; param_struct{2}.noiseM = 0;
param_struct{3} = param; param_struct{3}.noiseW = 0;

% this is for parallel running
parfor simulation_type = 1:3
    switch simulation_type
        % full noise model
        case simulation_type
            [output, ~] = place_cell_stochastic_update_snsm(gdInput,total_iter, param_struct{simulation_type});
            all_Yts{simulation_type} = output.Yt;
    end
end

%% 
blues = brewermap(11,'Blues');
reds = brewermap(11,'Reds');
greys = brewermap(11,'Greys');
fig_colors = [blues([7,11],:);reds([7,11],:);greys([7,11],:)];


time_points = round(total_iter/param.step);

% calculate the autocorrelation coefficients population vectors
% pvCorr = zeros(size(Yt,3),size(Yt,2)); 
% [~,neuroInx] = sort(peakInx(:,inxSel(1)));
    
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
    fh = shadedErrorBar((1:size(pvCorr,1))'*param.step,pvCorr',{@mean,@std});
    box on
    set(fh.edge,'Visible','off')
    fh.mainLine.LineWidth = 3;
    fh.mainLine.Color = fig_colors(2*phase-1,:);
    fh.patch.FaceColor = fig_colors(2*phase,:);
end

% ylim([0.25,1])
xlim([0,200]*param.step)
xlabel('Time','FontSize',16)
ylabel('PV correlation','FontSize',16)
set(gca,'FontSize',16,'LineWidth',1)


% save the figure
figPref = ['./figures',filesep,['1D_place_different_noise',date]];
saveas(gcf,[figPref,'.fig'])
print('-depsc',[figPref,'.eps'])


% save the data
dataFile = ['./data/revision',filesep,'1D_place_different_noise.mat'];
save(dataFile)
