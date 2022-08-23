% large scale simulation to check how the diffusion constant change with
% the noise level and eigen spectrum distribution
% this code should be ran on a cluster

type = 'psp';           % 'psp' or 'expansion
tau = 0.5;              %  scaling of the learning rate for M
learnType = 'online';  %  online, offline, batch   

% simulation type
simType = 'eigenSpectr';     % 'noiseAmp' or 'eigenSpectr'
totSimul = 2e3;              %  total number of sampling

%% Confifguration when running on cluster
% uncomment this section and modifiy the directory and settings
addpath('/n/home09/ssqin/driftRepr')
 
% start the parallel pool with 12 workers
parpool('local', str2num(getenv('SLURM_CPUS_PER_TASK')));

%% learning rate depedent, keep eigen spectrum and noise level, tau the same
n = 10;
k = 3;
rho = 0.95;             % the first k component explain the majority of the fluctuations      

num_sel = 200;          % randomly selected samples used to calculate the drift and diffusion constants
step = 10;              % store every 10 updates
tot_iter = 2e4;  
time_points = round(tot_iter/step);
Yt = zeros(k,time_points,num_sel);
learnRate = 0.08;       % learning rate
t = 10000;  % total number of samples

if strcmp(simType,'noiseAmp')
    repeats = 40;   % for statistical robustness
    noiseStd = ones(repeats,1)*10.^(-2:0.05:-1); 
    noiseStd = noiseStd(:);
    eigens = [4.5,3,1.5,ones(1,7)*0.01];
    allDiffConst = nan(length(noiseStd),2);  % store all the diffusion constant and exponents
    all_ave_rmsd = cell(length(noiseStd),1);  % store all the mean rmsd
    parfor i = 1:length(noiseStd)
        [dph,ephi,ave_rmsd] = rotaDiffConst(n,k, eigens, noiseStd(i),learnRate);
        allDiffConst(i,:) = [dph,ephi];
        all_ave_rmsd{i} = ave_rmsd;
        disp(noiseStd(i));
     end
elseif strcmp(simType,'eigenSpectr')
    noiseStd = 0.01; 
    eigens = nan(10,totSimul);        % store all the input eigen values
    allDiffConst = nan(totSimul,2) ;  %store all the diffusion constant and exponents
    all_ave_rmsd = cell(totSimul,1);  % store all the mean rmsd
    parfor i= 1:totSimul
        eigens(:,i) = SMhelper.genRandEigs(k,n,rho,'lognorm');
        [dph,ephi,ave_rmsd] = rotaDiffConst(n,k, eigens(:,i), noiseStd,learnRate);
        allDiffConst(i,:) = [dph,ephi];
        all_ave_rmsd{i} = ave_rmsd;
    end
end

% save the data for plot
dataFile = [simType,'.mat'];
save(dataFile,'noiseStd','eigens','allDiffConst','learnRate','step','all_ave_rmsd')