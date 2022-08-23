% This program re-fit the diffusion constants based on the average rmsd
noiseData = load('../data/noiseAmp_08232022.mat');  
Ds_linear = nan(length(noiseData.all_ave_rmsd),1);
for i = 1:length(noiseData.all_ave_rmsd)
    [dph_linear,~] = SMhelper.fitRotationDiff(noiseData.all_ave_rmsd{i},...
        noiseData.step,1000,'mean_linear');
    Ds_linear(i) = dph_linear;   
end

figure
plot(Ds_linear(:),noiseData.allDiffConst(:,1),'o')
set(gca,'XScale','log','YScale','log')


figure
plot(noiseData.learnRate*noiseData.noiseStd.^2./sum(noiseData.eigens(1:3).^2)/4,Ds_linear,'.')
set(gca,'XScale','log','YScale','log')
