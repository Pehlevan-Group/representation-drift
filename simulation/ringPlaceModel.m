% This program tests how the receptive field change in a 1D ring place cell
% model.
% This can also be used to  simulate T-maze task and PPC neurons


close all
clear

%% setting for the graphics
% plot setting
defaultGraphicsSetttings

rb = brewermap(11,'RdBu');
set1 = brewermap(9,'Set1');

saveFolder = ['.',filesep,'figures'];

%% Generate sample data, with small N

params.dim_out = 5;             % number of neurons
params.dim_in = 2;              % input dimensionality, 3 for Tmaze and 2 for ring
total_iter = 1e4;               % total simulation iterations

% default is a ring
dataType = 'ring';              % 'Tmaze' or 'ring';
learnType = 'snsm';             % snsm if using simple non-negative similarity matching
noiseVar = 'same';              % using different noise or the same noise level for each synpase
params.batch_size = 1;          % default 1
params.record_step = 1000;

% generate the ring data input, 2D
num_angle = 1e3;                % total number of angles
X = generate_ring_input(num_angle);

%% setup the learning parameters

params.noiseStd = 0.02;        % 0.005 for ring, 1e-3 for Tmaze
params.learnRate = 0.05;        % default 0.05

Cx = X*X'/size(X,2);            % input covariance matrix

% initialize the states
y0 = zeros(params.dim_out,params.batch_size);
Wout = zeros(1,params.dim_out); % linear decoder weight vector

params.W = 0.1*randn(params.dim_out,params.dim_in);
params.M = eye(params.dim_out); % lateral connection if using simple nsm
params.lbd1 = 0.0;              % regularization for the simple nsm, 1e-3
params.lbd2 = 0.02;             % default 1e-3

params.alpha = 0;               % should be smaller than 1 if for the ring model
params.beta = 1;                % 
params.gy = 0.05;               % update step for y
params.b = zeros(params.dim_out,1);      % bias

params.sigWmax = params.noiseStd;    % the maxium noise level for the forward matrix
params.sigMmax = params.noiseStd;    % maximum noise level of recurrent matrix

% assume uniform distribution at log scale
if strcmp(noiseVar, 'various')
%     noiseVecW = rand(k,1);
    noiseVecW = 10.^(rand(params.dim_out,1)*2-2);
    params.noiseW = noiseVecW*ones(1,params.dim_in)*params.sigWmax;   % noise amplitude is the same for each posterior 
%     noiseVecM = rand(k,1);
    noiseVecM = 10.^(rand(params.dim_out,1)*2-2);
    params.noiseM = noiseVecM*ones(1,params.dim_out)*params.sigMmax; 
else
    params.noiseW =  params.sigWmax*ones(params.dim_out,params.dim_in);   % stanard deivation of noise, same for all
    params.noiseM =  params.sigMmax*ones(params.dim_out,params.dim_out);   
end

% initial stage, make sure the weight matrices reach stationary state
[~, params] = ring_update_weight(X,total_iter,params);

% check the receptive field
Xsel = X(:,1:1:end);     % only use 10% of the data
Y0 = 0.1*rand(params.dim_out,size(Xsel,2));
Ys = nan(params.dim_out,size(Xsel,2));
for i = 1:size(Xsel,2)
    states_fixed_nn = MantHelper.nsmDynBatch(Xsel(:,i),Y0(:,i), params);
    Ys(:,i) = states_fixed_nn.Y;	
end

%% Continue with three different noise scenarios

% centroid postions
ampThd = 0.05;                     % threhold of place cell, 2/23/2021
flag = sum(Ys > ampThd,2) > 3;     % only those neurons that have multiple non-zeros values
RFctr0 = MantHelper.nsmCentroidRing(Ys,flag);

% ============ visualize the input ==============
visual_sep = 1;   % only plot very 5 data point
colors = flip(brewermap(round(num_angle/visual_sep),'Spectral'));
spectralMap = flip(brewermap(256,'Spectral'));

ringfig = figure;
set(ringfig,'color','w','Units','inches','Position',[0,0,3.2,2.8])
hold on
for i = 1:floor(num_angle/visual_sep)
    plot(X(1,i*visual_sep),X(2,i*visual_sep),'.','Color',colors(i,:),'MarkerSize',10)
end
hold off
colormap(spectralMap)
cbh = colorbar;
cbh.Ticks = linspace(0, 1, 3) ; %Create 8 ticks from zero to 1
cbh.TickLabels = {'0','\pi','2\pi'} ;
pbaspect([1 1 1])
set(gca,'XColor','none','YColor','none')


% ============= place field of neurons ==========
figure
imagesc(Ys)
% imagesc(states_fixed_nn.Y)
colorbar
xlabel('position','FontSize',28)
ylabel('neuron','FontSize',28)
ax = gca;
ax.XTick = [1 500 1000];
ax.XTickLabel = {'0', '\pi', '2\pi'};
set(gca,'FontSize',24)

figure
plot(Ys(1:4,:)')
% plot(states_fixed_nn.Y(1:4,:)')
% width of place fields
rfWidth = mean(Ys>0.01,2);
% rfWidth = mean(states_fixed_nn.Y>0.01,2);
figure
histogram(rfWidth)
xlabel('Width')
ylabel('Count')

% =============reorder the place field of neurons ==========
% sort the location
actiInx = sum(Ys>0.01,2) > 3;
Ya = Ys(actiInx,:);
[sortVal,sortedInx] = sort(Ya,2,'descend');
% [sortVal,sortedInx] = sort(states_fixed_nn.Y,2,'descend');
[~,neurOrder] = sort(sortedInx(:,1));

figure
imagesc(Ya(neurOrder,:))
colorbar
xlabel('position','FontSize',28)
ylabel('neuron','FontSize',28)
ax = gca;
ax.XTick = [1 500 1000];
ax.XTickLabel = {'0', '\pi', '2\pi'};
set(gca,'FontSize',24)


% distribution of peak value
figure
histogram(sortVal(:,1))
xlabel('Peak amplitude')
ylabel('Count')

%% continue updating with noise
params.record_step = 10;
total_iter = 1e4;    % number of conitnous iteration
[Yt, params] = ring_update_weight(X,total_iter,params);

%% analysis
% Estimate the diffusion constants
time_points = size(Yt,3);
centroidRF = nan(params.dim_out,time_points);
newPeakVals = nan(params.dim_out,time_points);
for i = 1:size(Yt,3)
    Ys_curr = Yt(:,:,i);
    
    flag = sum(Ys_curr > ampThd,2) > 3;     % only those neurons that have multiple non-zeros values
    centroidRF(:,i) = MantHelper.nsmCentroidRing(Ys_curr,flag);
    temp = ceil(centroidRF(:,i));
    for n = 1:params.dim_out
        if ~isnan(temp(n))
            newPeakVals(n,i) = Ys_curr(n,temp(n));
        end
    end
end

%%
newPeaks = centroidRF/max(centroidRF(:))*2*pi;     % 11/21/2020, in unit of rad

msds = nan(floor(time_points/2),params.dim_out);
for i = 1:floor(time_points/2)
    diffLag = min(abs(newPeaks(:,i+1:end,:) - newPeaks(:,1:end-i,:)),...
        2*pi - abs(newPeaks(:,i+1:end,:) - newPeaks(:,1:end-i,:)) );
    msds(i,:) = nanmean(diffLag.*diffLag,2);
end

% seperation of time between two data points
step = params.record_step*params.gy; 
Ds = PlaceCellhelper.fitLinearDiffusion(msds,step,'linear');
Ds_log = PlaceCellhelper.fitLinearDiffusion(msds,step,'log'); % different fitting methods


% diffusion constant vs synaptic noise
selNeur = ~isnan(Ds);
totNoiseStd = params.noiseW(:,1).^2 + params.noiseM(:,1).^2;

% linear regression
mdl = fitlm(totNoiseStd(selNeur),Ds(selNeur));

fit_stat = anova(mdl,'summary');
fh = plot(mdl);
fh(1).Marker = 'o';
fh(1).Color = 'k';
title('')
xlabel('$\sigma_W^2 + \sigma_M^2$','Interpreter','latex','FontSize',24)
ylabel('$D$','Interpreter','latex','FontSize',24)
set(gca,'YScale','linear','XScale','linear','FontSize',24)
figure
plot(totNoiseStd(selNeur), Ds(selNeur),'o','MarkerSize',10)
xlabel('$\sigma_w^2 + \sigma_M^2$', 'Interpreter','latex','FontSize',24)
ylabel('$D$', 'Interpreter','latex','FontSize',24)


% illustration of the msds
msdFig = figure;
set(msdFig,'color','w','Units','inches','Position',[0,0,3.2,2.8])
plot((1:length(msds))'*step,msds,'LineWidth',2)
xlabel('Time','FontSize',20)
ylabel('$\langle(\Delta \phi)^2\rangle$','Interpreter','latex','FontSize',20)
set(gca,'YTick',[0,pi/2,pi],'YTickLabel',{'0','\pi/2','\pi'},'XTick',...
    [1,5000,10000],'XTickLabel',{'0','5','10 \times 10^4'},'FontSize',16)

% save this figure
% prefix ='ring_msd_example';
% saveas(msdFig,[saveFolder,filesep,prefix,'.fig'])
% print('-depsc',[saveFolder,filesep,prefix,'.eps'])


nBins = size(Xsel,2);
figure
plot((1:size(centroidRF,2))*step,centroidRF(6:10,:)'/nBins*2*pi,'LineWidth',1)
xlabel('Time')
ylabel('Centroid of RF (rad)')

% figure
% plot((1:nBins)/nBins*2*pi,Yt(:,:,end)')
% xlabel('Time')
% ylabel('Response')

%% Analysis, check the change of place field
pkThreshold = 0.05;  % active threshold

% peak of receptive field
peakInx = nan(params.dim_out,time_points);
peakVal = nan(params.dim_out,time_points);
for i = 1:time_points
    [pkVal, peakPosi] = sort(Yt(:,:,i),2,'descend');
    peakInx(:,i) = peakPosi(:,1);
    peakVal(:,i) = pkVal(:,1);
end

% ======== faction of neurons have receptive field at a give time =====
% quantified by the peak value larger than a threshold 0.01
rfIndex = peakVal > pkThreshold;
tolActiTime = mean(rfIndex,2);

% fraction of neurons
activeRatio = sum(rfIndex,1)/params.dim_out;
figure
plot(101:time_points,activeRatio(101:end))
xlabel('iteration')
ylabel('active fraction')

figure
histogram(activeRatio(101:end))

% drop in 
figure
plot(sum(rfIndex,2)/params.dim_out)


% =========place field order by the begining ======================
inxSel = [100, 250, 400];
% inxSel = [100, 2000, 4000];
figure
for i = 1:length(inxSel)
    subplot(1,3,i)
%     imagesc(Yt(:,:,inxSel(i)))
    colorbar
    title(['iteration ', num2str(inxSel(i))])
    ax = gca;
    ax.XTick = [1 500 1000];
    ax.XTickLabel = {'0', '\pi', '2\pi'};
    ylabel('neuron index')
    xlabel('position')
end


% ======== ordered by current index ==========
figure
for i = 1:length(inxSel)
    [~,neuroInx] = sort(peakInx(:,inxSel(i)));
    subplot(1,3,i)
    imagesc(Yt(neuroInx,:,inxSel(i)))
    colorbar
    ax = gca;
    ax.XTick = [1 500 1000];
    ax.XTickLabel = {'0', '\pi', '2\pi'};
    title(['iteration ', num2str(inxSel(i))])
    xlabel('position')
    ylabel('sorted index')
end


% ======== representation similarity matrix =======
figure
for i = 1:length(inxSel)
%     [~,neuroInx] = sort(peakInx(:,inxSel(i)));
    SM = Yt(:,:,inxSel(i))'*Yt(:,:,inxSel(i));
    subplot(1,3,i)
    imagesc(SM,[0,3])
    colorbar
    ax = gca;
    ax.XTick = [1 500 1000];
    ax.YTick = [1 500 1000];
    ax.XTickLabel = {'0', '\pi', '2\pi'};
    ax.YTickLabel = {'0', '\pi', '2\pi'};
    title(['iteration', num2str(inxSel(i))])
    xlabel('position')
    ylabel('position')
end


% check the msd
figure
plot(msds(:,[1,2]))     
xlabel('time interva')
ylabel('MSD')


%% ========== Quantify the drifting behavior ==========================
% due to the periodic condition manifold, we need to correct the movement
% using the initial order as reference point

% orderedPeaks = peakInx(neurOrder,:);
orderedPeaks = pks(neurOrder,:);
L = size(Yt,2);
shift = diff(orderedPeaks')';
reCode = zeros(size(shift));
reCode(shift >= L/2) = -1;
reCode(shift < -L/2) = 1;
addVals = cumsum(reCode,2)*2*pi;
newPeaks = orderedPeaks/L*2*pi + [zeros(params.dim_out,1),addVals];
% newPeaks = orderedPeaks/1001*2*pi;   % 11/21/2020


%% 
save_folder = fullfile(pwd,'data', filesep,'revision');
save_data_name = fullfile(save_folder, ['ring_various_noise_',date,'.mat']);
save(save_data_name,'-v7.3')

%% For publication
sFolder = './figures';
figPre = ['ring_Batch_N200',date];

nc = 256;   % number of colors
spectralMap = brewermap(nc,'Spectral');
PRGnlMap = brewermap(nc,'PRGn');
% RdBuMap = flip(brewermap(nc,'RdBu'),1);

blues = brewermap(11,'Blues');
reds = brewermap(11,'Reds');
BuGns = brewermap(11,'BuGn');
greens = brewermap(11,'Greens');
greys = brewermap(11,'Greys');
PuRd = brewermap(11,'PuRd');
YlOrRd = brewermap(11,'YlOrRd');
oranges = brewermap(11,'Oranges');
RdBuMap = flip(brewermap(nc,'RdBu'),1);
BlueMap = flip(brewermap(nc,'Blues'),1);
GreyMap = brewermap(nc,'Greys');


% Parameters for plot
figWidth = 3.5;   % in unit of inches
figHeight = 2.8;
plotLineWd = 2;    %line width of plot
axisLineWd = 1;    % axis line width
labelFont = 20;    % font size of labels
axisFont = 16;     % axis font size

% ************************************************************
% Diffusion constant vs the synaptic noise level
% ************************************************************
% plot when we use various noise level for different neurons
if strcmp(noiseVar, 'various')
   
    % first loglog plot
    f_D_sig = figure;
    pos(3)=3.2; pos(4)=2.8;
    set(f_D_sig,'color','w','Units','inches','Position',pos)
    
    loglog(totNoiseStd(selNeur),Ds(selNeur),'o','MarkerSize',6,...
        'MarkerFaceColor',greys(1,:),'MarkerEdgeColor',greys(9,:),'LineWidth',1)
  
    
    xlabel('$\sigma_W^2 + \sigma_M^2$','Interpreter','latex','FontSize',16)
    ylabel('$D$','Interpreter','latex','FontSize',16)
    set(gca,'XTick',10.^(-5:-2))

    prefix = [figPre, 'D_sigmas'];
    saveas(f_D_sig,[sFolder,filesep,prefix,'.fig'])
    print('-depsc',[sFolder,filesep,prefix,'.eps'])
  
    % make a linear regression and plot the confidence interval and R^2
    sig_range = 10.^(-6:0.1:-3.8)';
    D_pred =  predict(mdl,sig_range);
    
    f_D_sig_linear = figure;
    set(f_D_sig_linear,'color','w','Units','inches','Position',pos)
    
    plot(totNoiseStd(selNeur),Ds(selNeur),'o','MarkerSize',6,...
       'MarkerEdgeColor',greys(9,:),'LineWidth',1.5)
    hold on
    plot(sig_range,D_pred,'LineWidth',2,'Color',blues(9,:))
    hold off
    xlabel('$\sigma_W^2 + \sigma_M^2$','Interpreter','latex','FontSize',16)
    ylabel('$D$','Interpreter','latex','FontSize',16)
    
    R2 = num2str(round(mdl.Rsquared.Ordinary*100)/100);
    pvalue = num2str(fit_stat.pValue(2));
    annotation('textbox', [.1 .8,.85  .1],...
    'String', ['R^2 =',R2,'; p-value =',pvalue],'BackgroundColor','none','Color','k',...
    'LineStyle','none','fontsize',10,'HorizontalAlignment','Center');

    prefix = [figPre, 'D_sigmas_linearRegr'];
    saveas(f_D_sig_linear,[sFolder,filesep,prefix,'.fig'])
    print('-depsc',[sFolder,filesep,prefix,'.eps'])
    
end


% ************************************************************
% Diffusion constant vs active time
% ************************************************************
% fit a linear regression model
mdl_D_acti = fitlm(tolActiTime(selNeur),Ds(selNeur));
D_pred = predict(mdl_D_acti,(0:0.1:1)');
fit_stat_D = anova(mdl_D_acti,'summary');
DiffuAmpFig = figure;
pos(3)=3.5;  pos(4)=2.8;
set(DiffuAmpFig,'color','w','Units','inches','Position',pos)
plot(tolActiTime(selNeur),Ds(selNeur),'o','MarkerSize',4,...
    'MarkerEdgeColor',greys(9,:),'LineWidth',1)
hold on
plot((0:0.1:1)',D_pred,'LineWidth',1.5,'Color',blues(9,:))
xlabel('Fraction of active time','FontSize',16)
ylabel('$D$','Interpreter','latex','FontSize',16)
set(gca,'FontSize',16,'LineWidth',1)

R2 = num2str(round(mdl_D_acti.Rsquared.Ordinary*100)/100);
pvalue = num2str(fit_stat_D.pValue(2));
annotation('textbox', [.1 .8,.85  .1],...
'String', ['R^2 =',R2,'; p-value =',pvalue],'BackgroundColor','none','Color','k',...
'LineStyle','none','fontsize',10,'HorizontalAlignment','Center');

prefix = [figPre, 'D_active_time'];
saveas(DiffuAmpFig,[sFolder,filesep,prefix,'.fig'])
print('-depsc',[sFolder,filesep,prefix,'.eps'])


% ************************************************************
% peak amplitude of example neuron
% ************************************************************
inx = 152;  % select one neuron to plot
temp = pkAmp(inx,:);
temp(isnan(temp)) = 0;
times = (1:length(temp))*step;   % iterations

f_pkAmpTraj = figure;
set(f_pkAmpTraj,'color','w','Units','inches')
pos(3)=3.2;  %17.4; %8.5;%pos(3)*1.5;
pos(4)=2.8;%pos(4)*1.5;
set(f_pkAmpTraj,'Position',pos)

plot(times',temp','LineWidth',2,'Color',blues(9,:))
xlabel('Time','FontSize',16)
ylabel('Peak Amplitude','FontSize',16)
set(gca,'LineWidth',1,'FontSize',16)

prefix = [figPre, 'pkAmp_example'];
saveas(f_pkAmpTraj,[sFolder,filesep,prefix,'.fig'])
print('-depsc',[sFolder,filesep,prefix,'.eps'])


% ************************************************************
% Fraction of active neurons
% ************************************************************
f_acti = figure;
set(f_acti,'color','w','Units','inches')
pos(3)=3.3;  
pos(4)=2.8;
set(f_acti,'Position',pos)
actiFrac = mean(pks > 0,1);

plot(times',actiFrac,'LineWidth',1.5)
xlabel('Time','FontSize',16)
ylabel('Active fraction','FontSize',16)
ylim([0,1])
set(gca,'LineWidth',1,'FontSize',16)

prefix = [figPre, 'fracActive'];
saveas(f_acti,[sFolder,filesep,prefix,'.fig'])
print('-depsc',[sFolder,filesep,prefix,'.eps'])


% ************************************************************
% average peak amplitude vs time of active
% ************************************************************
temp = pkAmp > ampThd;
tolActiTime = sum(temp,2)/size(pkAmp,2);
avePks = nan(size(pkAmp));
avePks(temp) = pkAmp(temp);
meanPks = nanmean(avePks,2);

f_ampActiTime= figure;
set(f_ampActiTime,'color','w','Units','inches')
pos(3)=3.5;  
pos(4)=2.8;
set(f_ampActiTime,'Position',pos)
plot(meanPks,tolActiTime,'o','MarkerSize',4,...
    'MarkerEdgeColor',greys(9,:),'LineWidth',1.5)
xlabel('Mean peak amplitude','FontSize',labelFont)
% ylabel({'faction of', 'active time'},'FontSize',labelFont)
ylabel(['faction of ', 'active time'],'FontSize',labelFont)
set(gca,'FontSize',axisFont,'LineWidth',axisLineWd)

prefix = [figPre, 'pkAmp_actiTime'];
saveas(f_ampActiTime,[sFolder,filesep,prefix,'.fig'])
print('-depsc',[sFolder,filesep,prefix,'.eps'])


% ************************************************************
% Diffusion constant and active period
% ************************************************************
DiffuAmpFig = figure;
pos(3)=3.5;  
pos(4)=2.8;
set(DiffuAmpFig,'color','w','Units','inches','Position',pos)
plot(tolActiTime,Ds(:,1),'o','MarkerSize',4,...
    'MarkerEdgeColor',greys(9,:),'LineWidth',1.5)
xlabel('Fraction of active time','FontSize',labelFont)
ylabel('$D$','Interpreter','latex','FontSize',labelFont)
set(gca,'FontSize',axisFont,'LineWidth',axisLineWd,'YScale','log')
% ylim([0,2.5])

prefix = [figPre, 'active_diffCont_'];
saveas(DiffuAmpFig,[sFolder,filesep,prefix,'.fig'])
print('-depsc',[sFolder,filesep,prefix,'.eps'])

% ************************************************************
% Diffusion constant vs averaged peak values
% ************************************************************
DiffuFig = figure;
pos(3)=3.5;  
pos(4)=2.8;
set(DiffuFig,'color','w','Units','inches','Position',pos)
plot(meanPks,Ds(:,1),'o','MarkerSize',4,...
    'MarkerEdgeColor',greys(9,:),'LineWidth',1.5)
xlabel('Mean peak amplitude','FontSize',labelFont)
ylabel('$D$','Interpreter','latex','FontSize',labelFont)
set(gca,'FontSize',axisFont,'LineWidth',axisLineWd,'YScale','log')
% ylim([0,2.5])

prefix = [figPre, 'ampl_diffCont_'];
saveas(DiffuFig,[sFolder,filesep,prefix,'.fig'])
print('-depsc',[sFolder,filesep,prefix,'.eps'])


% ************************************************************
% Single neuron drift illustration
% ************************************************************
singleNeuronFig = figure;
pos(3)=4.5;  
pos(4)=2.8;
set(singleNeuronFig,'color','w','Units','inches','Position',pos)
imagesc(squeeze(Yt),[0,1])
colorbar
xlabel('Time','FontSize',20)
ylabel('Position','FontSize',20)
set(gca,'YTick',[1,100,200],'YTickLabel',{'0', '\pi', '2\pi'},'FontSize',16)

prefix = 'singleNeuroDrift_';
saveas(singleNeuronFig,[sFolder,filesep,prefix,'.fig'])
print('-depsc',[sFolder,filesep,prefix,'.eps'])


%% Debug
% Distribution of nearest neighbor
nearestDist = nan(size(pks,2),2);  % store the mean and std
numActi = nan(size(pks,2),1);
L = size(Yt,2);
for i = 1:size(pks,2)
    temp = sort(pks(~isnan(pks(:,i)),i));
    ds = [diff(temp)/L*2*pi;(temp(1)-temp(end))/L*2*pi + 2*pi];
    nearestDist(i,:) = [mean(ds),var(ds)];
    numActi(i) = length(temp)/size(pks,1);
end

figure
hold on
plot((1:size(pks,2))', nearestDist(:,2))
plot([1;size(pks,2)],[2*pi/k;2*pi/k],'k--')
legend('variance')
box on
xlabel('Time')
ylabel('Distance')
title(['N = ',num2str(k)])

% correlation of centroid shift
dphi = diff(newPeaks,1,2);
inx = abs(dphi) > pi;
dphi(inx) = mod(dphi(inx),-sign(dphi(inx))*2*pi);

rho_phi = corr(newPeaks(1,:)',newPeaks(2,:)');
rho_dphi = corr(dphi(1,:)',dphi(2,:)');

% pairCorrCoef = nan(k*(k-1)/2,1);
temp = corrcoef(newPeaks');
m  = triu(true(size(temp)),1);
pairCorrCoef  = temp(m);

temp = corrcoef(dphi');
m  = triu(true(size(temp)),1);
dphiCorrCoef  = temp(m);



figure
plot((1:time_points)',newPeaks','LineWidth',1.5)
xlabel('Time','FontSize',24)
ylabel('$\phi$','Interpreter','latex','FontSize',24)
set(gca,'FontSize',20,'LineWidth',1.5)

figure
plot(dphi(1,:),dphi(3,:),'+','LineWidth',1.5)
legend(['\rho = ',num2str(dphiCorrCoef(2))],'Location','northwest')
xlabel('$\Delta\phi_1$','Interpreter','latex','FontSize',24)
ylabel('$\Delta\phi_2$','Interpreter','latex','FontSize',24)
set(gca,'FontSize',20,'LineWidth',1.5)


figure
plot(newPeaks(1,:),newPeaks(2,:),'.','LineWidth',1.5)
legend(['\rho = ',num2str(pairCorrCoef(1))],'Location','northwest')
xlabel('$\phi_1$','Interpreter','latex','FontSize',24)
ylabel('$\phi_2$','Interpreter','latex','FontSize',24)
set(gca,'FontSize',20,'LineWidth',1.5)


figure
bar(dphiCorrCoef,0.7)
xlabel('neuron pair')
ylabel('$\rho$','Interpreter','latex')
set(gca,'FontSize',20,'LineWidth',1.5)
title('correlation of $\Delta\phi$','Interpreter','latex')


figure
bar(pairCorrCoef,0.7)
xlabel('neuron pair')
ylabel('$\rho$','Interpreter','latex')
set(gca,'FontSize',20,'LineWidth',1.5)
title('correlation of $\phi$','Interpreter','latex')


interval = [10,100,1000];
for i = 1:length(interval)
    z = newPeaks(:,interval(i)+1:end) - newPeaks(:,1:end-interval(i));
    inx = abs(z) > pi;
    z(inx) = mod(z(inx),-sign(z(inx))*2*pi);

    rho = corr(z(1,:)',z(2,:)');
    figure
    plot(z(1,:),z(2,:),'.')
    legend(['\rho = ',num2str(rho)],'Location','northwest')
    xlabel('$\Delta\phi_1$','Interpreter','latex','FontSize',24)
    ylabel('$\Delta\phi_2$','Interpreter','latex','FontSize',24)
    set(gca,'FontSize',20,'LineWidth',1.5)
    title(['$\Delta t = ',num2str(interval(i)),'$'],'Interpreter','latex')

end

% example peak position before and after coninuous
figure
plot(newPeaks(1:3,:)','LineWidth',1.5)
set(gcf,'color','w')
xlabel('time')
ylabel('$\phi$','Interpreter','latex')
set(gca,'FontSize',20,'LineWidth',1.5)


figure
plot(pks(1:3,:)'/1000*2*pi,'LineWidth',2)
xlabel('time')
ylabel('$\phi$','Interpreter','latex')
set(gca,'FontSize',20,'LineWidth',1.5)

figure
plot(msds(:,1:3),'LineWidth',2)
xlabel('time')
ylabel('$\langle\Delta\phi^2\rangle$','Interpreter','latex')
set(gca,'FontSize',20,'LineWidth',1.5)


figure
plot(var(newPeaks,0,1))
xlabel('time')
ylabel('$\langle\phi(t)^2\rangle$','Interpreter','latex')


figure
histogram(dphi(1,:),'Normalization','pdf')
title('neuron 1')
xlabel('$\Delta\phi$','Interpreter','latex')
ylabel('pdf')
% xlim([-0.5,0.5])


% auto correlation function
figure
hold on
for i = 1:k
    acf = autocorr(newPeaks(i,:),'NumLags',2000);
    plot(acf)
end
hold off
box on
xlabel('$\Delta t$','Interpreter','latex')
ylabel('auto corr. coef.')
set(gca,'FontSize',20,'LineWidth',1.5)


% nearest neighor distance
nearestNeighor = nan(time_points,k);
normPks = peakInx/1001*2*pi;
for i = 1:k
    temp = normPks;
    temp(i,:) = [];
    rawDist = min(abs(temp -  normPks(i,:)),2*pi-abs(temp -  normPks(i,:)));
    dm = min(rawDist,[],1);
    nearestNeighor(:,i) = dm';
end

figure
hold on
% plot(nearestNeighor)
plot(mean(nearestNeighor,2))
plot([0;time_points],[pi/k;pi/k],'r-')
hold off
set(gcf,'color','w')
xlabel('Time')
ylabel({'Ave. Dist. to','nearest neighbor'})
box on
set(gca,'FontSize',20,'LineWidth',1.5)
ylim([0,2*pi/k])


% plot example RF at two time points
colors = brewermap(5,'Set1');
neuron_inx = [4,6,7,8,12];
Y1 =Yt(:,:,1)';
Y2 =Yt(:,:,50)';
figure
hold on
for i = 1:length(neuron_inx)
    plot((1:size(Y1,1))/size(Y1,1)*2*pi,Y1(:,neuron_inx(i)),'Color',colors(i,:))
    plot((1:size(Y2,1))/size(Y2,1)*2*pi,Y2(:,neuron_inx(i)),'--','Color',colors(i,:))
end
hold off
box on
xlabel('Position')
ylabel('Response')

%% Correlation of drift as a function of centroid distance

%newPeaks = mydata;
newPeaks = centroidRF/size(Xsel,2)*2*pi; % absolute position of centroid
theta_sep = pi/size(newPeaks,1);
dTheta = 0:theta_sep:pi;
shiftCentroid = cell(length(dTheta)-1,1);
deltaTau = 10;  % select time step seperation
self_shift = [];
repeats = min(deltaTau,20);  % number of repeats
aveShift = nan(length(dTheta)-1,repeats);

% a new method to estimate the correlations
for rp = 1:repeats
for bin = 1:length(dTheta)-1
    firstCen = [];
    secondCen = [];
    startIx = randperm(deltaTau);
    for i = startIx:deltaTau:(size(newPeaks,2) - deltaTau-startIx)
        % drift
        ds = newPeaks(:,i+deltaTau) - newPeaks(:,i);
        inx = abs(ds) > pi;
        ds(inx) = mod(ds(inx),-sign(ds(inx))*2*pi);  % real displacement

        % pairwise active centroid of these two time points
        centroidDist = squareform(mod(pdist(newPeaks(:,i)),pi));
        [ix,iy] = find(centroidDist > dTheta(bin) & centroidDist <= dTheta(bin+1));
        firstCen = [firstCen;ds(ix)];
        secondCen = [secondCen;ds(iy)];
    end
    % merge
    shiftCentroid{bin} = firstCen.*secondCen;
    aveShift(bin,rp) = nanmean(firstCen.*secondCen)/nanstd(firstCen)/nanstd(secondCen);
end
end

% avearge
aveRho = nan(length(dTheta),1);
for i = 1:length(dTheta)-1
    aveRho(i) = nanmean(shiftCentroid{i});
end

% plot the figure
figure
plot(dTheta,aveRho,'LineWidth',3)
xlabel('angular distance (rad)')
ylabel('covariance')
title(['N = ',num2str(size(newPeaks,1)),',$\delta t = ',num2str(deltaTau),'$'],'Interpreter','latex')


figure
hold on
plot(dTheta(2:end),aveShift,'LineWidth',1.5,'Color',greys(6,:))
plot(dTheta(2:end),mean(aveShift,2),'LineWidth',3,'Color',blues(8,:))
hold on
plot([0;pi],[0;0],'k--','LineWidth',1.5)
hold off
box on
xlabel('angular distance (rad)')
ylabel('$\rho$','Interpreter','latex')
title(['N = ',num2str(size(newPeaks,1)),',$\delta t = ',num2str(deltaTau),'$'],'Interpreter','latex')


%% For publication figure
% *********************************************
%  Position of centroids of two slected neurons
% *********************************************

% sel = randperm(k,2);
sel = [4,7];
centroidSel = pks(sel,1:500)'/1000*2*pi;
centroidFig = figure;
set(centroidFig,'color','w','Units','inches','Position',[0,0,3.2,2.8])
plot((1:size(centroidSel,1))'*step,centroidSel,'LineWidth',1.5)
xlim([0,5000])
xlabel('Time','FontSize',20)
ylabel('Centroid of RF','FontSize',20)
set(gca,'FontSize',16,'YTick',[0,pi,2*pi],'YTickLabel',{'0','\pi','2\pi'})

% save the figure
prefix ='ring_centroid_example';
saveas(centroidFig,[saveFolder,filesep,prefix,'.fig'])
print('-depsc',[saveFolder,filesep,prefix,'.eps'])


% ********************************************
%  Similarity matrix
% ********************************************

% sel = randperm(k,2);
Y1 = Yt(:,:,1);
Y2 = Yt(:,:,end);

smFig1 = figure;
set(smFig1,'color','w','Units','inches','Position',[0,0,3.5,2.9])
imagesc(Y1'*Y1)
colorbar
title('$t=1$','Interpreter','latex','FontSize',20)
xlabel('Position','FontSize',20)
xlabel('Position','FontSize',20)
set(gca,'FontSize',16,'YTick',[1,500,1001],'YTickLabel',{'0','\pi','2\pi'},...
    'XTick',[1,500,1001],'XTickLabel',{'0','\pi','2\pi'})

prefix ='ring_sm_example_t1';
saveas(smFig1,[saveFolder,filesep,prefix,'.fig'])
print('-depsc',[saveFolder,filesep,prefix,'.eps'])


smFig2 = figure;
set(smFig2,'color','w','Units','inches','Position',[0,0,3.5,2.9])
imagesc(Y2'*Y2)
colorbar
title('$t=2\times 10^4$','Interpreter','latex','FontSize',20)
xlabel('Position','FontSize',20)
xlabel('Position','FontSize',20)
set(gca,'FontSize',16,'YTick',[1,500,1001],'YTickLabel',{'0','\pi','2\pi'},...
    'XTick',[1,500,1001],'XTickLabel',{'0','\pi','2\pi'})
prefix ='ring_sm_example_t2';
saveas(smFig2,[saveFolder,filesep,prefix,'.fig'])
print('-depsc',[saveFolder,filesep,prefix,'.eps'])


% ********************************************************
% Also plot the population activity
% ********************************************************
[~,sortedInx] = sort(Y1,2,'descend');
[~,neurOrder] = sort(sortedInx(:,1));

pv1_fig = figure;
set(pv1_fig,'color','w','Units','inches','Position',[0,0,3.5,2.9])
imagesc(Y1(neurOrder,:),[0,0.5])
colorbar
ax = gca;
ax.XTick = [1 500 1000];
ax.XTickLabel = {'0', '\pi', '2\pi'};
ylabel('neuron index')
xlabel('position')

pv2_fig = figure;
set(pv2_fig,'color','w','Units','inches','Position',[0,0,3.5,2.9])
imagesc(Y2(neurOrder,:),[0,0.5])
colorbar
ax = gca;
ax.XTick = [1 500 1000];
ax.XTickLabel = {'0', '\pi', '2\pi'};
ylabel('neuron index')
xlabel('position')
%}