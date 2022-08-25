function out = PSP_noise_data_aggregate(noiseData,type)
% aggregate the statistics of diffusion constant
% input is a data struct and scaling factor correct the D estimation

out = [];  % store the output into a struct
if strcmp(type,'offline')
    num_std = floor(length(noiseData.noiseStd)/40);  % number of different std
    selInx = 1:num_std;
    syn_noise_std = reshape(noiseData.noiseStd,[40,num_std]);
    out.noiseAmpl = syn_noise_std(1,:).^2;
    allD_noise = noiseData.allDiffConst(:,1)/4;  %  the factor 4 is due to fitting
    out.aveD_noise = mean(reshape(allD_noise,[40,num_std]),1);
    out.stdD_noise = std(reshape(allD_noise,[40,num_std]),0,1);
    out.prefForm = sum(1./noiseData.eigens(1:3).^2)*noiseData.learnRate/2;
    out.D_pred = out.noiseAmpl(selInx)'*out.prefForm;
elseif strcmp(type,'online')
    num_std = floor(length(noiseData.noiseStd)/40);  % number of different std
    selInx = 1:num_std;
    syn_noise_std = reshape(noiseData.noiseStd,[40,num_std]);
    out.noiseAmpl = syn_noise_std(1,:).^2;
    allD_noise = noiseData.allDiffConst(:,1);
    out.aveD_noise = mean(reshape(allD_noise,[40,num_std]),1);
    out.stdD_noise = std(reshape(allD_noise,[40,num_std]),0,1);
    out.prefForm = sum(1./noiseData.eigens(1:3).^2)*noiseData.learnRate/4;
    out.D_pred = out.noiseAmpl(selInx)'*out.prefForm;
elseif strcmp(type, 'refit')
    num_std = floor(length(noiseData.noiseStd)/40);  % number of different std
    selInx = 1:num_std;
    syn_noise_std = reshape(noiseData.noiseStd,[40,num_std]);
    out.noiseAmpl = syn_noise_std(1,:).^2;
    out.aveD_noise = mean(reshape(noiseData.Ds_new,[40,num_std]),1);
    out.stdD_noise = std(reshape(noiseData.Ds_new,[40,num_std]),0,1);
    out.prefForm = sum(1./noiseData.eigens(1:3).^2)*noiseData.learnRate/4;
    out.D_pred = out.noiseAmpl(selInx)'*out.prefForm;
else
    disp('must be either offline or online!')
end
    out.num_std = num_std;
    out.selInx = selInx;
end


