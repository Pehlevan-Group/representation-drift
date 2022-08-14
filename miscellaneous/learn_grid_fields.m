% this program learns grid cells from place cell input

close all
clear

% generate plce fields
dim = 1;  % 1D environment, or 2D
param.num_fields = 200;
param.ps = 628;           % spatial postion sampled on each dimension
param.field_width = pi/4; % each dimension is parameterized to 2pi
param.num_grid = 10;      % number of grid cells

place_fields = generate_place_fields(dim, param.field_width, ...
    param.num_fields, param.ps);

% initialize the weight matrices
param.W = 0.05*rand(param.num_grid, param.num_fields);
param.M = eye(param.num_grid);
param.b = zeros(param.num_grid,1);
param.learn_rate = 0.05;
param.noise_std = 1e-2;
param.noiseW = param.noise_std*ones(param.num_grid, param.num_fields);
param.noiseM = param.noise_std*ones(param.num_grid, param.num_grid);

param.alpha = 10;  
param.lbd1 = 0.02;
param.lbd2 = 0.05;  
param.step = 10;       % recording steps
param.gy = 0.05;       % this is the integration time step in Euler rule
param.BatchSize = 1;   % online learning


param.batch_size = 1;  % default online learning

total_iter = 5e2;


[Yt, param] = grid_cell_stochastic_update_snsm(place_fields,total_iter, param);


%% 
% PCA of input
[COEFF, SCORE, LATENT, TSQUARED, EXPLAINED] = pca(place_fields');