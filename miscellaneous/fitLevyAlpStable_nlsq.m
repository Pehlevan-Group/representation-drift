function MSE = fitLevyAlpStable_nlsq(mu)
% this function return the MSE between the ECDF of two distribution
% interaction matrix is sampled from x, typcially the mean and std

% define global variables
global param targetData targetECDF x_eval
% mult = 1;
% co = DKL_kNN_k_initialization(mult);
% co = DKL_expF_initialization(mult);
% co = DKL_vME_initialization(mult);

% generate shift based on the distribution of step size
N  = length(targetData);
ds =  stblrnd(mu(1),0,mu(2),0,N,1);
ds(ds>1)=1;
ds(ds<-1) = -1;

rRaw = ds.*param.L + targetData;  % new position

inx1 = rRaw > param.L;
rRaw(inx1) = max(2*param.L - rRaw(inx1),0);
inx2 = rRaw < 0;
rRaw(inx2) = min(abs(rRaw(inx2)),param.L);

% shfit of centroid
dr = (rRaw - targetData)/param.L;

% x = -0.9:0.01:0.9;  % we don't need fit all the range
% [targetECDF, x] = ecdf(oneDayShift/param.L);
% predECDF = stblcdf(x,mu(1),0,mu(2),0);
[rwECDF, xrw] = ecdf(dr);
% x_eval = 0.1:0.005:0.9;
yrw = nan(length(x_eval),1);
for i = 1:length(x_eval)
    if x_eval(i) > max(xrw)
        yrw(i) = 1;
    elseif x_eval(i) < min(xrw)
        yrw(i) = 1;
    else
        yrw(i) = max(rwECDF(xrw<=x_eval(i)));
    end
end



MSE = norm(yrw - targetECDF);

figure
hold on
plot(x_eval,targetECDF)
plot(x_eval,yrw)
hold off
% 
% figure
% plot(targetECDF(:), yrw(:),'o')

end