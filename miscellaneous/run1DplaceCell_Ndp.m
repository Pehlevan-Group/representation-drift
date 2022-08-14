% diffusion for small Np

close all
clear


Nps = [2, 5,10];
% Nps = [1,2];

reps = 20;  % repeats for 10 tims
allDs = cell(length(Nps),1);
for i = 1:length(Nps)
    Ds = nan(Nps(i),reps);
    for j = 1:reps
        Ds(:,j) = placeCell_1Diff(Nps(i),0.01,0,0.05,15,2e4,0.01,1,'snsm');
    end
    allDs{i} = Ds;
end

% save the data
sFile = './data/pcDiffSmallNp_1D.mat';
save(sFile)