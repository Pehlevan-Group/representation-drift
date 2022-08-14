% plot 1D place cell results: simulation and modeling, Fig 5

%%
dFile = '../data/1Dplace_slice_0701.mat';
load(dFile)

% define all the colors
sFolder = ['..',filesep,'figures'];
figPre = ['placeCell_1D_slice_N200_',date];

%%
noiseVar  = 'same';
% ************************************************************
% Diffusion constant vs the synaptic noise level
% ************************************************************
% plot when we use various noise level for different neurons
if strcmp(noiseVar, 'various')
   
    % first loglog plot
    f_D_sig = figure;
    pos(3)=3.2; pos(4)=2.8;
    set(f_D_sig,'color','w','Units','inches','Position',pos)
    
    loglog(tot_noise_var(eff_acti_inx),Ds(eff_acti_inx),'o','MarkerSize',6,...
        'MarkerFaceColor',greys(1,:),'MarkerEdgeColor',greys(9,:),'LineWidth',1)
  
    
    xlabel('$\sigma_W^2 + \sigma_M^2$','Interpreter','latex','FontSize',16)
    ylabel('$D$','Interpreter','latex','FontSize',16)
    set(gca,'XTick',10.^(-7:-4))

    prefix = [figPre, 'D_sigmas'];
    saveas(f_D_sig,[sFolder,filesep,prefix,'.fig'])
    print('-depsc',[sFolder,filesep,prefix,'.eps'])
  
    % make a linear regression and plot the confidence interval and R^2
    sig_range = 10.^(-7:0.1:-4)';
    D_pred =  predict(mdl,sig_range);
    
    f_D_sig_linear = figure;
    set(f_D_sig_linear,'color','w','Units','inches','Position',pos)
    
    plot(tot_noise_var(eff_acti_inx),Ds(eff_acti_inx),'o','MarkerSize',6,...
       'MarkerEdgeColor',greys(9,:))
    hold on
    plot(sig_range,D_pred,'LineWidth',1.5,'Color',blues(9,:))
    hold off
    xlabel('$\sigma_W^2 + \sigma_M^2$','Interpreter','latex','FontSize',16)
    ylabel('$D$','Interpreter','latex','FontSize',16)
    
    R2 = num2str(round(mdl.Rsquared.Ordinary*100)/100);
    pvalue = num2str(fit_stat.pValue(2));
    annotation('textbox', [.1 .8,.85  .1],...
    'String', ['R^2 =',R2,'; p-value =',pvalue],'BackgroundColor','none','Color','k',...
    'LineStyle','none','fontsize',10,'HorizontalAlignment','Center');
% 
%     prefix = [figPre, 'D_sigmas_linearRegr'];
%     saveas(f_D_sig_linear,[sFolder,filesep,prefix,'.fig'])
%     print('-depsc',[sFolder,filesep,prefix,'.eps'])
    
end


% ************************************************************
% Diffusion constant vs active time
% ************************************************************
% fit a linear regression model
eff_acti_inx = ~isnan(Ds);
mdl_D_acti = fitlm(tolActiTime(eff_acti_inx),Ds(eff_acti_inx));
D_pred = predict(mdl_D_acti,(0:0.1:1)');
fit_stat_D = anova(mdl_D_acti,'summary');
DiffuAmpFig = figure;
pos(3)=3.5;  pos(4)=2.8;
set(DiffuAmpFig,'color','w','Units','inches','Position',pos)
plot(tolActiTime(eff_acti_inx),Ds(eff_acti_inx),'o','MarkerSize',4,...
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

% prefix = [figPre, 'D_active_time'];
% saveas(DiffuAmpFig,[sFolder,filesep,prefix,'.fig'])
% print('-depsc',[sFolder,filesep,prefix,'.eps'])

% ************************************************************
% peak position, using time point 1
% ************************************************************
temp = [ceil(pks(:,1)/param.ps),mod(pks(:,1),param.ps)]/param.ps;
pkVec = temp(~isnan(temp(:,1)),:);

% define figure size
f_pkPosi = figure;
set(f_pkPosi,'color','w','Units','inches')
pos(3)=3.2;  %17.4; %8.5;%pos(3)*1.5;
pos(4)=2.8;  % pos(4)*1.5;
set(f_pkPosi,'Position',pos)
% histogram(activeInter(:))
% xlim([0,100])
plot(pkVec(:,1),pkVec(:,2),'o','MarkerSize',4,'MarkerFaceColor',greys(9,:),...
    'MarkerEdgeColor',greys(9,:))
xlabel('X position','FontSize',16)
ylabel('Y position','FontSize',16)
set(gca,'LineWidth',1,'FontSize',16)

% prefix = [figPre, 'pkPosi'];
% saveas(f_pkPosi,[sFolder,filesep,prefix,'.fig'])
% print('-depsc',[sFolder,filesep,prefix,'.eps'])



% ************************************************************
% peak amplitude of example neuron
% ************************************************************
inx = 34;  % select one neuron to plot
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
% 
% prefix = [figPre, 'pkAmp'];
% saveas(f_pkAmpTraj,[sFolder,filesep,prefix,'.fig'])
% print('-depsc',[sFolder,filesep,prefix,'.eps'])


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
xlabel('t','Interpreter','latex','FontSize',20)
ylabel('Active fraction','FontSize',20)
ylim([0,1])
xlim([0,4000])
set(gca,'LineWidth',1,'FontSize',16)

% prefix = [figPre, 'fracActive'];
% saveas(f_acti,[sFolder,filesep,prefix,'.fig'])
% print('-depsc',[sFolder,filesep,prefix,'.eps'])

%%

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

%%

% Diffusion constant and active period
DiffuAmpFig = figure;
pos(3)=3.5;  
pos(4)=2.8;
set(DiffuAmpFig,'color','w','Units','inches','Position',pos)
plot(tolActiTime,Ds(:,1),'o','MarkerSize',4,...
    'MarkerEdgeColor',greys(9,:),'LineWidth',1.5)
xlabel('Fraction of active time','FontSize',labelFont)
ylabel('$D$','Interpreter','latex','FontSize',labelFont)
set(gca,'FontSize',axisFont,'LineWidth',axisLineWd)
ylim([0,2.5])

% prefix = [figPre, 'active_diffCont_'];
% saveas(DiffuAmpFig,[sFolder,filesep,prefix,'.fig'])
% print('-depsc',[sFolder,filesep,prefix,'.eps'])
% 


%%
% ************************************************************
% Change of population vectors, measured by correlation
% ************************************************************

% fmSel = randperm(size(Yt,2),1); % randomly slect three frames to consider
pvCorr = zeros(size(Yt,3),size(Yt,2)); 
% [~,neuroInx] = sort(peakInx(:,inxSel(1)));

for i = 1:size(Yt,3)
    for j = 1:size(Yt,2)
        temp = Yt(:,j,i);
        C = corrcoef(temp,Yt(:,j,1));
        pvCorr(i,j) = C(1,2);
    end
end

f_pvCorr = figure;
set(f_pvCorr,'color','w','Units','inches')
pos(3)=figWidth;  
pos(4)=figHeight;
set(f_pvCorr,'Position',pos)
fh = shadedErrorBar((1:size(pvCorr,1))',pvCorr',{@mean,@std});
box on
set(fh.edge,'Visible','off')
fh.mainLine.LineWidth = 3;
fh.mainLine.Color = blues(10,:);
fh.patch.FaceColor = blues(7,:);
% ylim([0.25,1])
xlim([0,400])
xlabel('Time','FontSize',16)
ylabel('PV correlation','FontSize',16)
set(gca,'FontSize',16,'LineWidth',1)

% prefix = [figPre, 'pvCorrCoef'];
% saveas(f_pvCorr,[sFolder,filesep,prefix,'.fig'])
% print('-depsc',[sFolder,filesep,prefix,'.eps'])

%%
% ************************************************************
% Representational similarity across time
% ************************************************************
% for better viusalization, only use part of the data
% gray color maps
% selInx = 1:10:1024;
selInx = find(~isnan(pks(:,1)));

% Y1= Yt(selInx,:,1);
% Y2 = Yt(selInx,:,1000);
Y1= Yt(:,:,1000);
Y2 = Yt(:,:,1300);


SM1 = Y1'*Y1;
SM2 = Y2'*Y2;
% SM1 = Y1(:,selInx)'*Y1(:,selInx);
% SM2 = Y2(:,selInx)'*Y2(:,selInx);

f_sm1 = figure;
set(f_sm1,'color','w','Units','inches')
pos(3)=3;  
pos(4)=3;
set(f_sm1,'Position',pos)

imagesc(SM1,[0,15]);
% colormap(GreyMap)
% cb = colorbar;
% set(cb,'FontSize',12)
set(gca,'XTick','','YTick','','LineWidth',0.5,'Visible','off')

% prefix = [figPre, 'sm1'];
% saveas(f_sm1,[sFolder,filesep,prefix,'.fig'])
% print('-depsc',[sFolder,filesep,prefix,'.eps'])

f_sm2 = figure;
set(f_sm2,'color','w','Units','inches')
pos(3)=3;  
pos(4)=3;
set(f_sm2,'Position',pos)

imagesc(SM2,[0,15]);
% colormap(GreyMap)
% cb = colorbar;
% set(cb,'FontSize',12)
set(gca,'XTick','','YTick','','LineWidth',0.5,'Visible','off')
% prefix = [figPre, 'sm2'];
% saveas(f_sm2,[sFolder,filesep,prefix,'.fig'])
% print('-depsc',[sFolder,filesep,prefix,'.eps'])


% heatmap of population vector odered by Y1
[pkSorted,sortedInx] = sort(Y1,2,'descend');
% significant nonzeor at time 1
actiTime1Inx = find(pkSorted(:,1)>0.05);

% [~,neurOrder] = sort(sortedInx(:,1));
[~,neurOrder] = sort(sortedInx(actiTime1Inx,1));



pv1Fig = figure;
pos(3)=3;  
pos(4)=3;
set(pv1Fig,'color','w','Units','inches','Position',pos)
% imagesc(Y1(neurOrder(250:end),:),[0,2])
imagesc(Y1(actiTime1Inx(neurOrder),:),[0,3])
set(gca,'Visible','off')
colorbar
% prefix = [figPre, 'pvHeatMap1'];
% saveas(pv1Fig,[sFolder,filesep,prefix,'.fig'])
% print('-depsc',[sFolder,filesep,prefix,'.eps'])


pv2Fig = figure;
pos(3)=3;  
pos(4)=3;
set(pv2Fig,'color','w','Units','inches','Position',pos)
imagesc(Y2(actiTime1Inx(neurOrder),:),[0,3])
% imagesc(Y2(neurOrder(250:end),:),[0,2])
set(gca,'Visible','off')
colorbar

% prefix = [figPre, 'pvHeatMap2_'];
% saveas(pv2Fig,[sFolder,filesep,prefix,'.fig'])
% print('-depsc',[sFolder,filesep,prefix,'.eps'])

%% Probability of shift of peak positions

psRange = param.ps;   % the range of ring positions
pks0 = pkCenterMass;
tps = [50,100,200,400,500,800];
quantiles = (1:3)/4*psRange;
probs = nan(length(tps),length(quantiles));
for i = 1:length(tps)
    diffLag = abs(pks0(:,tps(i)+1:end,:) - pks0(:,1:end-tps(i),:));
    for j = 1:length(quantiles)
        probs(i,j) = sum(diffLag(:)>quantiles(j))/length(diffLag(:));
    end
end


probShift = figure; 
set(probShift,'color','w','Units','inches')
pos(3)=3.8;  %17.4; %8.5;%pos(3)*1.5;
pos(4)=2.8;%pos(4)*1.5;
set(probShift,'Position',pos)
hold on
for i = 1:length(quantiles)
    plot(tps'*step,probs(:,i),'o-','MarkerSize',8,'Color',blues(1+3*i,:),...
        'MarkerFaceColor',blues(1+3*i,:),'MarkerEdgeColor',blues(1+3*i,:),'LineWidth',1.5)
end
hold off
box on

lg = legend('\Delta s >1/4 L','\Delta s>1/2 L','\Delta s >3/4 L');
set(lg,'FontSize',14)
xlabel('$\Delta t$','Interpreter','latex','FontSize',20)
ylabel('Probability','FontSize',20)
set(gca,'LineWidth',1,'FontSize',16)

% prefix = 'Tmaze_prob_shift_dist';
% saveas(gcf,[sFolder,filesep,prefix,'.fig'])
% print('-depsc',[saveFolder,filesep,prefix,'.eps'])

%% Statistics of shift of RF, Compare with experiment
% This part is intended to compare with ﻿Gonzalez et al Science 2019 paper
% based on unsorted data

psRange = param.ps;  %total length of the linearized track

% tps = [400,800,1500];
tps = [200,400,700];


quantiles = (1:3)/4*param.ps;
probs = nan(length(tps),length(quantiles));
edges = -200:10:200;
numBins = length(edges)-1; % this is the same as in the experiments

fractionSep = nan(length(tps),numBins);  % store the fraction of different shift
randSiftFrac = nan(length(tps),numBins);  % store randomized shift

shifDist = figure; 
set(shifDist,'color','w','Units','inches')
pos(3)=figWidth;  %17.4; %8.5;%pos(3)*1.5;
pos(4)=figHeight;%pos(4)*1.5;
set(shifDist,'Position',pos)
hold on
for i = 1:length(tps)
    temp = pkCenterMass(:,tps(i)+1:end,:) - pkCenterMass(:,1:end-tps(i),:);
    
    [f,xi] = ksdensity(temp(~isnan(temp))); 
%     histogram(temp(~isnan(temp)),'Normalization','pdf')
    plot(xi,f,'Color',blues(1+3*i,:),'LineWidth',2)
%     diffLag = min(pi, abs(temp(~isnan(temp))));
    diffLag = abs(temp(~isnan(temp)));
    
   [fractionSep(i,:),~]= histcounts(temp(~isnan(temp)),edges,'Normalization','probability');
    
    for j = 1:length(quantiles)
        probs(i,j) = sum(diffLag(:)>quantiles(j))/length(diffLag(:));
    end
end

% compared with random position
randPks = nan(size(pkCenterMass));
for i = 1:size(pkCenterMass,2)
    inx = find(~isnan(pkCenterMass(:,i)));
    randPks(inx,i) = psRange*rand(length(inx),1);
end
temp = pkCenterMass - randPks;
diffLagRnd = abs(temp(~isnan(temp)));
[f,xi] = ksdensity(temp(~isnan(temp))); 


% randomized shift
for i=1:length(tps)
    temp = cell(1,1);
    inx = find(~isnan(pkCenterMass(:,tps(i))));
    for j = 1:50
        rps = nan(size(pkCenterMass,1),1);
        rps(inx) = psRange*rand(length(inx),1);
        temp{j} = pkCenterMass(:,tps(i)) - rps;
    end
    rndSf = cat(1,temp{:});
    [randSiftFrac(i,:),~] = histcounts(rndSf(~isnan(rndSf)),edges,'Normalization','probability');
end

% [randSiftFrac,~] = histcounts(temp(~isnan(temp)),numBins,'Normalization','probability');
plot(xi,f,'Color',reds(8,:),'LineWidth',2)
hold off
box on
% xlim([-6.5,6.5])
lg = legend(['\Delta t=',num2str(tps(1)*step)],['\Delta t=',num2str(tps(2)*step)],...
    ['\Delta t=',num2str(tps(3)*step)],'Random','Location','northeast');
set(lg,'FontSize',14)
xlabel('$\Delta s$','Interpreter','latex','FontSize',24)
ylabel('pdf','FontSize',24)
set(gca,'LineWidth',1,'FontSize',20)


% Histount of the probability
histFig = figure;
pos(3)=figWidth; 
pos(4)=figHeight;
set(histFig,'color','w','Units','inches','Position',pos)

hold on
for i = 1:length(tps)
    plot(fractionSep(i,:)','Color',blues(1+3*i,:),'LineWidth',plotLineWd)
    plot(randSiftFrac(i,:)','Color',reds(1+3*i,:),'LineWidth',plotLineWd)
end
box on
lg = legend(['\Delta t=',num2str(tps(1)*step)],['\Delta t=',num2str(tps(2)*step)],...
    ['\Delta t=',num2str(tps(3)*step)],'Random');
set(lg,'FontSize',14)
% xlim([0,50])
xticks([1,(1:4)*numBins/4])
x = -1:0.5:1;
xticklabels({x});
xlabel('Centroid shift','FontSize',labelFont)
ylabel('Fraction','FontSize',labelFont)
set(gca,'FontSize',axisFont)

% save the figure
% prefix = [figPre, 'shiftDistrFrac'];
% saveas(histFig,[sFolder,filesep,prefix,'.fig'])
% print('-depsc',[sFolder,filesep,prefix,'.eps'])