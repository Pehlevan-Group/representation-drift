% This program re-fit the diffusion constants based on the average rmsd
clear
% load('../data/PSP_offline/eigenSpectr_offline_0824.mat');
% load('../data/PSPonline/eigenSpectr_online_0825_005.mat');
load('../data/eigenSpectr_08232022_1.mat');


reFitMthd = 'linear';
fitRange = 200;

Ds_new = nan(length(all_ave_rmsd),1);
for i = 1:length(all_ave_rmsd)
    [dph_linear,~] = SMhelper.fitRotationDiff(all_ave_rmsd{i},...
        step,fitRange,reFitMthd);
    Ds_new(i) = dph_linear;   
end

% add a new fields to existing data struct


% figure
% plot(Ds_log(:),noiseData.allDiffConst(:,1),'o')
% set(gca,'XScale','log','YScale','log')


% for noise dependence
% figure
% plot(learnRate*noiseStd.^2*sum(1./eigens(1:3).^2)'/4,Ds_new(:),'.')
% set(gca,'XScale','log','YScale','log')

% for eigenspectrum dependence
figure
plot(learnRate*noiseStd.^2*sum(1./eigens(1:3,:).^2,1)'/4,Ds_new(:),'.')
set(gca,'XScale','log','YScale','log')

% save('../data/PSP_offline/eigenSpectr_offline_0824.mat');
% save('../data/eigenSpectr_08232022_1.mat')
% save('../data/PSPonline/eigenSpectr_online_0825_005.mat');

figure
hold on
plot(learnRate*noiseStd.^2*sum(1./eigens(1:3,:).^2,1)'/4,allDiffConst(:,1),'.')
set(gca,'XScale','log','YScale','log')


% example rmsd
% rnd_sel = randperm(2000,10);
% figure
% hold on
% for i = 1:10
%     plot(all_ave_rmsd{rnd_sel(i)})
% end
% hold off
% set(gca,'XScale','log','YScale','log')
