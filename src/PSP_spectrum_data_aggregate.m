function out = PSP_spectrum_data_aggregate(spectrumData,type)
% aggregate the statistics of diffusion constant
% input is a data struct and scaling factor correct the D estimation

out = [];  % store the output
allEigs = spectrumData.eigens(1:3,:);
sumKeigs = sum(1./(allEigs.^2),1);
if strcmp(type,'offline')
    allD_spectr = spectrumData.allDiffConst(:,1)/4;     % factor 4 in fitting
else
    allD_spectr = spectrumData.allDiffConst(:,1);
end

nBins = 20;
spRange = 10.^[-0.5,2];     % this range depends on the data set used
dbin = (log10(spRange(2)) - log10(spRange(1)))/nBins;
out.aveSpDs = nan(nBins, 2);  % average and standard deviation
for i = 1:nBins
    inx = log10(sumKeigs) >= log10(spRange(1)) + (i-1)*dbin & log10(sumKeigs) < log10(spRange(1)) + i*dbin;
    out.aveSpDs(i,:) = [mean(allD_spectr(inx)),std(allD_spectr(inx))];
end
out.centers = spRange(1)*10.^(dbin*(1:nBins));

% a linear fit based on the averaged diffusion constant
selInx = 1:20;      % remove the last two data points

if strcmp(type,'offline')
    out.theoPre = spectrumData.learnRate*spectrumData.noiseStd^2/2; % factor 2 or 4
else
    out.theoPre = spectrumData.learnRate*spectrumData.noiseStd^2/4; % factor 2 or 4
end


end


