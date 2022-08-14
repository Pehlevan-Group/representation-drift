function runRingModeDriftComp(Np,sig,lr,alp, bs,iter,repeats)
% repeat the simulation for each parameter set
% Data used to generate figure 4 and related SI figures

if Np  > 1
    allCentroidDist = cell(repeats,1);
    allAveActi = nan(repeats,1);
    for i = 1:repeats
        [nearestDist,numActi] = ringModelCompRW(Np,sig,lr,alp, bs,iter);
        allCentroidDist{i} = nearestDist;
        allAveActi(i) = nanmean(numActi);
    end
else
    allPks = cell(repeats,1);  % store the peak position when Np = 1
    for i = 1:repeats
        [~,~,pks] = ringModelCompRW(Np,sig,lr,alp, bs,iter);
        allPks{i} = pks;
    end
end

% save the data
sFile = ['./data/pcRing_CentroidRW_alp',num2str(alp),'_Np',num2str(Np),...
    '_std',num2str(sig),'_lr',num2str(lr),'_bs',num2str(bs),'.mat'];
save(sFile, '-v7.3')
end