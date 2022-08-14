% diffusion small Np
% close all
% clear


Nps = [1,2,4,8,16];
% Nps = [1,2];
noiseStd = 0.01;
alp = 95;
lbd1 = 0.0;
lbd2 = 0.05;
learnRate = 0.05;
tot_iter = 2e4;
rwStep = 1;
lnTy = 'snsm';

reps = 40;  % repeats for 10 tims
allDs = cell(length(Nps),1);
for i = 1:length(Nps)
    Ds = nan(Nps(i),reps);
    for j = 1:reps
%         Ds(:,j) = placeCellClusterDiff(Nps(i),0,0,0,85,2e4,j);
        Ds(:,j) = placeCellClusterDiff(Nps(i),noiseStd,lbd1,lbd2,alp,tot_iter,learnRate,rwStep, lnTy);
    end
    allDs{i} = Ds;
end

% save the data
% sFile = './data/pcDiffSmallNp_randwalk.mat';
sFile = '/n/home09/ssqin/representationDrift/pc2D_smallN_0219.mat';
save(sFile)