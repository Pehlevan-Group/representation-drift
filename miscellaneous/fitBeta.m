function H = fitBeta(x)
% this function return KL divergency 

% define global variables
global param targetData oneDayShift 
mult = 1;
% co = DKL_kNN_kiTi_initialization(mult);
% co = DKL_PSD_SzegoT_initialization(mult);
co = DKL_expF_initialization(mult);
% co = DKL_vME_initialization(mult);

% generate shift based on the distribution of step size
N  = length(targetData);
% ds = exprnd(mu,N,1);
ds = betarnd(x(1),x(2),N,1);
% mask1 = rand(N,1) > param.rho;  %  shift larger than 0
mask2 = sign(randn(N,1));
rRaw = ds.*mask2*param.L + targetData; % new position

inx1 = abs(rRaw) > param.L;
rRaw(inx1) = min(2*param.L - rRaw(inx1),param.L);
inx2 = abs(rRaw) < 0;
rRaw(inx2) = min(abs(rRaw(inx2)),param.L);

% shfit of centroid
dr = rRaw - targetData;

nonZerosDr = dr(dr~=0);
% compare the distribution by calculating the K-L divergence
% H = DKL_PSD_SzegoT_estimation(nonZerosDr'/param.L,oneDayShift'/param.L,co);
% H = DKL_vME_estimation(nonZerosDr'/param.L,oneDayShift'/param.L,co);
H = DKL_expF_estimation(nonZerosDr'/param.L,oneDayShift'/param.L,co);

% figure
% hold on
% histogram(nonZerosDr'/param.L,'Normalization','pdf')
% histogram(oneDayShift'/param.L,'Normalization','pdf')
% hold off
end