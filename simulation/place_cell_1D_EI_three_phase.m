% using non-negative similarity matching to learn a 1-d place cells based
% on predefined grid cells patterns. Considering both excitatory and
% inhibitory neurons. Explore how different noise sources contribute to the
% drift. Wei, Wie and M are "recurrent matrices"


clear
close all

%% model parameters
param.ps =  200;        % number of positions along each dimension
param.Nlbd = 5;         % number of different scales/spacing
param.Nthe = 6;         % number of rotations
param.Nx =  4;          % offset of x-direction
param.Ny = 4;           % offset of y-direction
param.Ng = param.Nlbd*param.Nthe*param.Nx*param.Ny;   % total number of grid cells
param.Np = 100;         % number of place cells, default 20*20
param.Nin = 20;         % number of inhibitory interneurons

param.baseLbd = 1/4;    % spacing of smallest grid RF, default 1/4
param.sf =  1.42;       % scaling factor between adjacent module

noiseStd = 0.05;        % 0.001
learnRate = 0.02;       % default 0.05

param.W = 0.5*randn(param.Np,param.Ng);      % initialize the forward matrix
param.Wei = 0.05*rand(param.Np,param.Nin);   % i to e connection
param.Wie = param.Wei';                      % e to i connections
param.M = eye(param.Nin);                    % recurrent interaction between inhibiotry neurons
param.lbd1 = 0.01;                           % 0.15 for 400 place cells and 5 modes
param.lbd2 = 0.05;                           % 0.05 for 400 place cells and 5 modes

param.alpha = 60;       % the threshold depends on the input dimension
param.beta = 10; 
param.gy = 0.05;        % update step for y
param.gz = 0.05;        % update step for z
param.b = zeros(param.Np,1);  % biase
param.learnRate = learnRate;  % learning rate for W and b
param.noise =  noiseStd;      % stanard deivation of noise 
param.rwSpeed = 10;           % steps each update, default 1
param.step = 10;        % storage step

param.ori = 1/10*pi;          % slicing orientation

BatchSize = 1;          % minibatch used to to learn
learnType = 'snsm';     % snsm, batch, online, randwalk, direction, inputNoise
noiseVar = 'same';      % same noise level for all neurons


param.sigWmax = noiseStd;% the maximum noise std of forward synapses
param.sigMmax = noiseStd;     
if strcmp(noiseVar, 'various')
    noiseVecW = 10.^(rand(param.Np,1)*2-2);
    param.noiseW = noiseVecW*ones(1,param.Ng)*param.sigWmax;   % noise amplitude is the same for each posterior 
    noiseVecM = 10.^(rand(param.Nin,1)*2-2);
    param.noiseM = noiseVecM*ones(1,param.Nin)*param.sigMmax;
    noiseVecWei = 10.^(rand(param.Np,1)*2-2);
    param.noiseWei = noiseVecWei*ones(1,param.Nin)*param.sigMmax;
    noiseVecWie = 10.^(rand(param.Nin,1)*2-2);
    param.noiseWie = noiseVecWei*ones(1,param.Np)*param.sigMmax;
else
    param.noiseW =  param.sigWmax*ones(param.Np,param.Ng);     % stanard deivation of noise, same for all
    param.noiseWei =  param.sigWmax*ones(param.Np,param.Nin);
    param.noiseWie =  param.sigWmax*ones(param.Nin,param.Np);
    param.noiseM =  param.sigMmax*ones(param.Nin,param.Nin);   
end

param.ori = 1/10*pi;     % slicing orientation

param.BatchSize = 1;      % minibatch used to to learn
param.learnType = 'snsm';  % snsm, batch, online, randwalk, direction, inputNoise

gridQuality = 'slice';  % regular, weak or slice

makeAnimation = 0;    % whether make a animation or not

gridFields = slice2Dgrid(param.ps,param.Nlbd,param.Nthe,param.Nx,param.Ny,param.ori);


%% using non-negative similarity matching to learn place fields
% generate input from grid filds

total_iter = 1e4;   % total interation, default 2e3

posiGram = eye(param.ps);
gdInput = gridFields'*posiGram;  % gramin of input matrix

% The initial stage, only return the updated parameters
[~, param] = place_cell_EI_update_snsm(gdInput,total_iter, param);

%% Continuous Nosiy update
% we will have three different scenarios
% full noise, either W or M
total_iter = 1e4;
all_Yts = cell(3,1);        % store all the ensemnbles
all_Zts = cell(3,1);        % store inhibitory neurons
PV_corr_coefs = cell(3,1);  % store the population vector correlations, principal neurons
PV_corr_coefs_inhi = cell(3,1); % inhibitory neurons
param_struct = cell(3,1);
param_struct{1} = param;
param_struct{2} = param; param_struct{2}.noiseM = 0;
param_struct{2}.noiseWie = 0;param_struct{2}.noiseWei = 0;
param_struct{3} = param; param_struct{3}.noiseW = 0;

% this is for parallel running
parfor simulation_type = 1:3
    switch simulation_type
        % full noise model
        case simulation_type
            [output, ~] = place_cell_EI_update_snsm(gdInput,total_iter, param_struct{simulation_type});
            all_Yts{simulation_type} = output.Yt;
            all_Zts{simulation_type} = output.Zt;
    end
end

%% 
blues = brewermap(11,'Blues');
reds = brewermap(11,'Reds');
greys = brewermap(11,'Greys');
fig_colors = [blues([7,11],:);reds([7,11],:);greys([7,11],:)];


time_points = round(total_iter/param.step);


%  for excitatory neurons
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
xlim([0,500]*param.step)
xlabel('Time','FontSize',16)
ylabel('PV correlation','FontSize',16)
set(gca,'FontSize',16,'LineWidth',1)


% save the figure
figPref = ['./figures',filesep,['1D_place_different_noise_PV_corr_exci',date]];
saveas(gcf,[figPref,'.fig'])
print('-depsc',[figPref,'.eps'])


% save the data
% dataFile = ['./data/revision',filesep,'1D_place_different_noise.mat'];
% save(dataFile)

%% For inhibitory neurons
f_pvCorr_I = figure;
set(f_pvCorr_I,'color','w','Units','inches')
pos(3)=3.5;  
pos(4)=2.8;
set(f_pvCorr_I,'Position',pos)

for phase = 1:3
    pvCorr_Inhi = zeros(size(all_Zts{phase},3),size(all_Zts{phase},2)); 
    for i = 1:size(all_Zts{phase},3)
        for j = 1:size(all_Zts{phase},2)
            temp = all_Zts{phase}(:,j,i);
            C = corrcoef(temp,all_Zts{phase}(:,j,1));
            pvCorr_Inhi(i,j) = C(1,2);
        end
    end
    PV_corr_coefs_inhi{phase} = pvCorr_Inhi;
    % plot
    fh = shadedErrorBar((1:size(pvCorr_Inhi,1))'*param.step,pvCorr_Inhi',{@mean,@std});
    box on
    set(fh.edge,'Visible','off')
    fh.mainLine.LineWidth = 3;
    fh.mainLine.Color = fig_colors(2*phase-1,:);
    fh.patch.FaceColor = fig_colors(2*phase,:);
end

% ylim([0.25,1])
xlim([0,500]*param.step)
xlabel('Time','FontSize',16)
ylabel('PV correlation','FontSize',16)
set(gca,'FontSize',16,'LineWidth',1)


% save the figure
figPref = ['./figures',filesep,['1D_place_different_noise_PV_corr_inihi',date]];
saveas(gcf,[figPref,'.fig'])
print('-depsc',[figPref,'.eps'])
