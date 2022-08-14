function H = fitLevyAlpStable(mu)
% this function return differential entropy 
% interaction matrix is sampled from x, typcially the mean and std

% define global variables
global param targetData oneDayShift 
mult = 1;
% co = DKL_kNN_k_initialization(mult);
co = DKL_expF_initialization(mult);
% co = DKL_vME_initialization(mult);

% generate shift based on the distribution of step size
N  = length(targetData);
% alp = mu(1);
% gam = mu(2);
ds =  stblrnd(mu(1),0,mu(2),0,N,1);
ds(ds>1)=1;
ds(ds<-1) = -1;
% mask1 = rand(N,1) > param.rho;  %  shift larger than 0
% mask2 = sign(randn(N,1));
rRaw = ds.*param.L + targetData;  % new position

inx1 = rRaw > param.L;
rRaw(inx1) = max(2*param.L - rRaw(inx1),0);
inx2 = rRaw < 0;
rRaw(inx2) = min(abs(rRaw(inx2)),param.L);

% shfit of centroid
dr = rRaw - targetData;

% nonZerosDr = dr(dr~=0);
drSel = dr(abs(dr) < 0.9*param.L);  % we want emphasize the center part of the distri
% drSel = dr;  % we want emphasize the center part of the distri.

% compare the distribution by calculating the K-L divergence
% H = DKL_vME_estimation(nonZerosDr'/param.L,oneDayShift'/param.L,co);
H = DKL_expF_estimation(drSel'/param.L,oneDayShift'/param.L,co);
% H = DKL_kNN_k_estimation(nonZerosDr'/param.L,oneDayShift'/param.L,co);

% figure
% hold on
% histogram(dr'/param.L,'Normalization','pdf')
% histogram(oneDayShift'/param.L,'Normalization','pdf')
% hold off

end