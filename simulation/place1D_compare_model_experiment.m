% this program compares place fields drift in our model and in the 
% in our model and in the experiment
% The experimental data is from Gonzalez et al, Science 2019
% 1) Decay of correlation coefficient vs time
% 2) Distribution of centroid shift
% 3) Fraction of active place cell vs time
% 4) Distance 

% load the data
% sFolder = '/Users/shawnqin/OneDrive - Harvard University/MATLAB/representationDrift/figures';

%% load the summarized data
dFolder = '/Users/shawnqin/Dropbox (Harvard University)/analyzedData';
dFile1 = 'hippocampus_data_1222.mat';   % without information of active cells
dFile2 = 'hippocampus_data.mat';        % without information of active cells


load(fullfile(dFolder,filesep,dFile1))
load(fullfile(dFolder,filesep,dFile2))
% dFile = 'hippocampus_data.mat';   % without information of active cells

%% Graphics setting and colors

defaultGraphicsSetttings

nc = 256;   % number of colors in color map
spectralMap = brewermap(nc,'Spectral');
PRGnlMap = brewermap(nc,'PRGn');
RdBuMap = flip(brewermap(nc,'RdBu'),1);
BlueMap = flip(brewermap(nc,'Blues'),1);

blues = brewermap(11,'Blues');
reds = brewermap(11,'Reds');
RdBu = brewermap(11,'RdBu');
BuGns = brewermap(11,'BuGn');
greens = brewermap(11,'Greens');
greys = brewermap(11,'Greys');
PuRd = brewermap(11,'PuRd');
YlOrRd = brewermap(11,'YlOrRd');
oranges = brewermap(11,'Oranges');
set1 = brewermap(11,'Set1');

labelFontSize = 20; % lable font size
errSymSize = 4;     % symbol size
lineWd = 2;         % line width of plot
axisLw = 1;         % line width of axis
axisFont = 16;      % axis font size


%% Align based on day 1

dayRef = 1;
ind0 = zeros(1);
allOrderPV = cell(size(allCat1,1),1);
numberShared = nan(size(allCat1,1),1);
PVcorr = nan(size(allCat1,1),2);
%covert digit to register ID, first number is animal number '0000' is a
%spacer then followed by the cell ID.
for i=1:size(allCat1{dayRef,2},1)
    ind0(i,1) = str2num(strcat(num2str(allCat1{dayRef,2}(i,1)),'0000',num2str(allCat1{dayRef,2}(i,2))));
end

figSM = figure;
set(gcf,'color','w','Units','inches','Position',[0,0,15,2])
for day=1:size(allCat1,1)
    ind = zeros(1); 
    for i=1:size(allCat1{day,2},1)
        ind(i,1) = str2num(strcat(num2str(allCat1{day,2}(i,1)),'0000',num2str(allCat1{day,2}(i,2))));  
    end
    [~,sharedIx,sharedIx0] = intersect(ind,ind0,'stable'); 
    allOrderPV{day} = allCat1{day,2}(sharedIx,4:end);
    numberShared(day) = size(allOrderPV{day},1);
    
    % PV correlation coefficient
    ce = nan(size(allCat1{day,8},2),1); % store the Pearson correlation coeff
    for ps = 4:size(allCat1{day,8},2)
        C = corrcoef(allCat1{day,2}(sharedIx,ps),allCat1{dayRef,2}(sharedIx0,ps));
        ce(ps) = C(1,2);
    end
    PVcorr(day,:) = [nanmean(ce),nanstd(ce)];
    
    
    h(day) = subplot(1,size(allCat1,1),day);
    imagesc(allOrderPV{day}'*allOrderPV{day},[0 130])
    set(gca,'Visible','off')
end

% use to eliminate blank edges
for day=1:length(h)
    set(h(day),'position',[(day-1)/length(h) 0 1/length(h) 1])
end

% fraction of number of shared neurons
neuronOL = nan(size(allCat1,1),1);
days = [1:5,16:20]';
% figure
% plot(days,numberShared/numberShared(dayRef),'o-','MarkerSize',10,'LineWidth',3)
% xlabel('days')
% ylabel('neuron overlap')


% population vector correlation coefficients
% figure
% errorbar(days,PVcorr(:,1),PVcorr(:,2),'o-','MarkerSize',errSymSize,'MarkerFaceColor',blues(9,:),...
%     'MarkerEdgeColor',blues(9,:),'Color',blues(9,:),'LineWidth',lineWd)
% xlabel('Days','FontSize',labelFontSize)
% ylabel('PV correlation','FontSize',labelFontSize)
% set(gca,'FontSize',axisFont,'LineWidth',axisLw)
% 

%% Example representation and similarity matrix
day1 = 1;
day2 = 10;

% find the shared neurons that have active RF
ind = zeros(1); 
for i=1:size(allCat1{day2,2},1)
    ind(i,1) = str2num(strcat(num2str(allCat1{day2,2}(i,1)),'0000',num2str(allCat1{day2,2}(i,2))));  
end
[~,sharedIx0,sharedIx] = intersect(ind0,ind,'stable'); 

Y1 = allCat1{day1,2}(sharedIx0,4:end);
Y2 = allCat1{day2,2}(sharedIx,4:end);

% figure
% subplot(1,2,1)
% imagesc(Y1)
% 
% subplot(1,2,2)
% imagesc(Y2)
% 
% % simiarity matrix
% figure
% subplot(1,2,1)
% imagesc(Y1'*Y1,[0,80])
% 
% subplot(1,2,2)
% imagesc(Y2'*Y2,[0,80])


%% Shift of place fields
%{
tmp = cell(1);
tmpR = cell(1);
binWidth = 50;
% figure('pos',[200 200 1200 400])
% subplot(1,2,1)
figure
%field drift in real data
k=1;
dayApart = [1,10,20];
for i = 1:length(dayApart)
    for animal = 1:size(centroidShift_P,1)
        if size(centroidShift_P{animal, 1},1)>dayApart(i)
            tmp{k}(animal,:) = centroidShift_P{animal, 1}(dayApart(i),:);
        end
    end
    errorbar(median(tmp{k},1),std(tmp{k},1)./size(centroidShift_P,1),'-o',...
        'MarkerSize',6,'Color',blues(2+3*i,:),'MarkerEdgeColor',blues(2+3*i,:),...
        'MarkerFaceColor',blues(2+3*i,:),'CapSize',0,'LineWidth',1.5)
    hold on
    k=k+1;
end 

k=1;
%field drift in random data
for i = 1:length(dayApart)
    for animal = 1:size(centroidShift_P,1)
        if size(centroidShift_P{animal, 1},1)>dayApart(i)
            tmpR{k}(animal,:) = centroidShift_P_R{animal, 1}(dayApart(i),:);
        end
    end
    errorbar(median(tmpR{k},1),std(tmpR{k},1)./size(centroidShift_P,1),'-',...
        'MarkerSize',3,'Color',greys(2+3*i,:),'MarkerEdgeColor',greys(2+3*i,:),...
        'MarkerFaceColor',greys(2+3*i,:),'CapSize',0,'LineWidth',1.5)
%     hold on
    k=k+1;
end 
legend('1 day','10 days','20 days')
ylim([0 0.20])
xlim([1 binWidth])
xticks(1:5:binWidth+1)
x = -binWidth/2:5:binWidth/2;
xticklabels({x});
xlabel('Centroid shift')
ylabel('Fraction')
set(gca,'FontSize',20)
%}
%% shift of centroid, use more data

daysep = 50;                   % total interval of day seperation
shiftMean = nan(daysep,binWidth);     % store the mean and standard deviation
shiftStd = nan(daysep,binWidth);
shiftMeanR = nan(daysep,binWidth);     % store mean of randomly shuffled field
shiftStdR = nan(daysep,binWidth);
% dayApart = [1,10,20];
tmp = cell(1);
tmpR = cell(1);
k = 1;
for day = 1:daysep
    for animal = 1:size(centroidShift_P,1)
        if size(centroidShift_P{animal, 1},1)>day
            tmp{k}(animal,:) = centroidShift_P{animal, 1}(day,:);
        end
        
        if size(centroidShift_P_R{animal, 1},1)>day
            tmpR{k}(animal,:) = centroidShift_P_R{animal, 1}(day,:);
        end
    end
    shiftMean(day,:) = mean(tmp{k},1);
    shiftStd(day,:) = std(tmp{k},0,1);
    
    shiftMeanR(day,:) = mean(tmpR{k},1);
    shiftStdR(day,:) = std(tmpR{k},0,1);

    k=k+1;
end 


%% Correlation of drift as a function of centroid distance

%newPeaks = mydata;
trackL = 50;  % length of track
centroid_sep = 0:5:trackL;   % bins of centroid positions
deltaTau = 1;  % select time step seperation
numAnimals = size(pc_Centroid.allCentroids,1);
shiftCentroid = cell(length(centroid_sep)-1,numAnimals);
aveShift = nan(length(centroid_sep)-1,numAnimals);
oneDayShift = cell(size(pc_Centroid.allCentroids,1),1);  % store all the one day shifts 

oneDayShiftAdj = cell(size(pc_Centroid.allCentroids,1),1);  % after adjusting boundary effect
oneDayShiftRm = cell(size(pc_Centroid.allCentroids,1),1);   % after removing two ends

rmL = 10;  % number of bins removed from the two ends

% Distribution of all the one-day centroid shift
for animal = 1:size(pc_Centroid.allCentroids,1)
    Z = pc_Centroid.allCentroids{animal};
    Z(Z==0) = nan;  %
    dayInters = dayInterval{animal}(:,1);
    dayInx = find(diff(dayInters)==1)+1;
    cenShift = [];  % store for one animal
    cenShiftAdj = []; % adjusted 
    cenShiftRm = [];  % removed two ends
    for i = 1:length(dayInx)
        bothActiInx = ~isnan(Z(:,dayInx(i))) & ~isnan(Z(:,dayInx(i)-1));
        ds = Z(bothActiInx,dayInx(i)) - Z(bothActiInx,dayInx(i)-1);
        cenShift = [cenShift;ds];  % this is the absolute shift
        
        % consider the refrectory boundary effect, only when they appear
        % inside the two ends regions
        baNeurons = find(bothActiInx);
        leftRegion = Z(bothActiInx,dayInx(i))<rmL & Z(bothActiInx,dayInx(i)-1) < rmL;
        refract = rand(length(leftRegion),1) <=0.5 & leftRegion;
        if any(refract)
            ds(refract) = -Z(baNeurons(refract),dayInx(i)) - Z(baNeurons(refract),dayInx(i)-1);  % refractory boundary
        end
        
        RightRegion =  Z(bothActiInx,dayInx(i))>= trackL-rmL & Z(bothActiInx,dayInx(i)-1) > trackL-rmL;
        refract = rand(length(RightRegion),1) <=0.5 & RightRegion;
        if any(refract)
            ds(refract) = 2*trackL - Z(baNeurons(refract),dayInx(i)) - Z(baNeurons(refract),dayInx(i)-1);
        end
        cenShiftAdj = [cenShiftAdj;ds];
        
        % Yet another way to define the distributon, only conider the
        % centorid within the middle 60% of the track
        selNeurons = Z(bothActiInx,dayInx(i)-1) >= rmL & Z(bothActiInx,dayInx(i)-1) <= trackL-rmL;
        ds = Z(baNeurons(selNeurons),dayInx(i)) - Z(baNeurons(selNeurons),dayInx(i)-1);
        cenShiftRm = [cenShiftRm;ds];
        
    end
    oneDayShift{animal} = cenShift;
    
    oneDayShiftAdj{animal} = cenShiftAdj;
    oneDayShiftRm{animal} = cenShiftRm;
end

% plot the distribution of one day centroid shift
mergeShifts = cat(1,oneDayShift{:});
figure
histogram(mergeShifts/50,'Normalization','pdf')
xlabel('Centroid shift (L)')
ylabel('Probability')

% merge adjust one-day shift
mergeShiftsAdj =  cat(1,oneDayShiftAdj{:});

% merge removed centroids one-day shift
mergeShiftsRm =  cat(1,oneDayShiftRm{:});
figure
histogram(mergeShiftsRm/50,'Normalization','pdf')
xlabel('Centroid shift (L)')
ylabel('Probability')
title('removed')

% a new method to estimate the correlations
rmL = 10;   % remove frist and last 10 bins
for animal = 1:size(pc_Centroid.allCentroids,1)
    for bin = 1:length(centroid_sep)-1
        firstCen = [];
        secondCen = [];
        Z = pc_Centroid.allCentroids{animal};
        Z(Z==0) = nan;  %
        dayInters = dayInterval{animal}(:,1);
        dayInx = find(diff(dayInters)==1)+1;
        for i = 1:length(dayInx)
            
            bothActiInx = ~isnan(Z(:,dayInx(i))) & ~isnan(Z(:,dayInx(i)-1));
            ds = Z(bothActiInx,dayInx(i)) - Z(bothActiInx,dayInx(i)-1);
            temp = Z(bothActiInx,dayInx(i)-1);
            
            % remove first 5 and last 5 bins
            rmInx = temp < rmL | temp > 50-rmL;
            temp(rmInx) = nan;
            
            % pairwise active centroid of these two time points
            centroidDist = squareform(pdist(temp));
            [ix,iy] = find(centroidDist > centroid_sep(bin) & centroidDist <= centroid_sep(bin+1));
            firstCen = [firstCen;ds(ix)];
            secondCen = [secondCen;ds(iy)];
        end
        % merge
        shiftCentroid{bin,animal} = firstCen.*secondCen;
        aveShift(bin,animal) = nanmean(firstCen.*secondCen)/nanstd(firstCen)/nanstd(secondCen);
    end
end

% avearge
aveRho = nan(length(centroid_sep)-1,numAnimals);
for am = 1:numAnimals
    for i = 1:length(centroid_sep)-1
        aveRho(i,am) = nanmean(shiftCentroid{i,am});
    end
end
finalAveRho = nanmean(aveRho,2);


% *********************************************
% Plot distance-dependent correlation of drift
% *********************************************
figure
hold on
plot(centroid_sep(2:end)'/trackL,aveShift,'Color',greys(7,:),'LineWidth',2)
plot(centroid_sep(2:end)'/trackL,nanmean(aveShift,2),'Color',blues(7,:),'LineWidth',4)
hold off
box on
% xlim([5,45])
xlabel('Distance (L)')
ylabel('$\rho$','Interpreter','latex')
% figPref = [sFolder,filesep,'hipp_shift_corr_coef_rm'];
% saveas(gcf,[figPref,'.fig'])
% print('-depsc',[figPref,'.eps'])


% plot the distribution of centroid
anm = 13;   % select one animal to plot
Z = pc_Centroid.allCentroids{anm};
Z(Z==0) = nan;  %
ix = any(~isnan(Z),2);
Zsel = Z(ix,:);
dayInters = dayInterval{anm}(:,1);

% another way to represent the centroid position
actiDays = sum(~isnan(Zsel),2);
[~,neuInx] = sort(actiDays,'descend');
exampleNeurons = neuInx(1:3);   % example neurons to high light the centroids
exampleColors = [blues(8,:);oranges(4,:);PuRd(6,:)];


% *********************************************
% Visualize the shift of example RFs
% *********************************************
figure
set(gcf,'renderer','painters');
hold on
for i = 1:length(dayInters)
    % first a horizontal reference line
    plot([0;50],ones(2,1)*dayInters(i),'Color',greys(6,:),'LineWidth',1)
    
    % add centroids at each day
    cts = Zsel(~isnan(Zsel(:,i)),i);
    Xs = ones(2,1)*cts';
    Ys = [dayInters(i)-0.3;dayInters(i)+0.3]*ones(1,length(cts));
    plot(Xs,Ys,'Color',greys(11,:),'LineWidth',1)
    
    % highlight thress example neurons
    xe = ones(2,1)*Zsel(exampleNeurons,i)';
    ye = [dayInters(i)-0.3;dayInters(i)+0.3]*ones(1,length(exampleNeurons));
    for j = 1:length(exampleNeurons)
        plot(xe(:,j),ye(:,j),'Color',exampleColors(j,:),'LineWidth',3)
    end
end
hold off
box on
% ylim([-0.5,13])
xlim([2,48])
xlabel('Position (bin)')
ylabel('Day')

% figPref = [sFolder,filesep,'hipp_centroid_posit_animal13'];
% saveas(gcf,[figPref,'.fig'])
% print('-depsc',[figPref,'.eps'])


%% compare with  model on the distance-dependent drift
% the random walk step is generated from experimental distribution
% we can either use the experimental data as the step size
% or use the fitted exponential distributed step size
% the estimated pdf of onestep shift is from "fitStepSize_1D.m"
trackL = 50;
repeats = 20;
numNeu = 100;
totSteps = 20;
mergeShifts = mergeShiftsRm;   % if we adjust the experimental shift
rwType = 'experiment';  % or 'experiment'
muOpt = 0.1722;        % this is from "fitStepSize_1D.m"

Ns = length(mergeShifts);
centroids_model = cell(repeats,1); % store all the centroid shift

for rp = 1:repeats
    centroids_model{rp} = nan(numNeu,totSteps);
    for i = 1:numNeu
        temp = randi(trackL,1);
        if strcmp(rwType, 'experiment')
            ds = randperm(length(mergeShifts),1);  % random generate a shift
        elseif strcmp(rwType, 'model')
            ds = exprnd(muOpt)*trackL*sign(randn); % generate a step size using exponential dist. model
        end
            
        for j = 1:totSteps
            newTry = temp + mergeShifts(randperm(Ns,1));
%             newTry = temp + randn*gaussStd;
            if newTry < 0
                temp = abs(newTry);
            elseif newTry > trackL
                temp = max(1,2*trackL - newTry);  % can't be negative
            else
                temp = newTry;
            end            

            centroids_model{rp}(i,j) = temp;
        end
    end
end

% estimated the distance-dependent correlation of centroid shift
% centroid_sep = 0:1:trackL;   % bins of centroid positions
deltaTau = 1;      % select time step seperation, subsampling

shiftCentroid = cell(length(centroid_sep)-1,repeats);
aveShiftRW = nan(length(centroid_sep)-1,repeats);
% allCov  = nan(length(centroid_sep)-1,repeats);

rmL = 10;  % number of bins removed from the two ends
for rp = 1:repeats
    initialTime = 1;
    for bin = 1:length(centroid_sep)-1
        firstCen = [];
        secondCen = [];

        for i = initialTime:deltaTau:(size(centroids_model{rp},2)-deltaTau)

            bothActiInx = ~isnan(centroids_model{rp}(:,i+deltaTau)) & ~isnan(centroids_model{rp}(:,i));
            ds = centroids_model{rp}(bothActiInx,i+deltaTau) - centroids_model{rp}(bothActiInx,i);
            temp = centroids_model{rp}(bothActiInx,i);
            
            % remove two ends
            rmInx = temp < rmL | temp > trackL - rmL;
            temp(rmInx) = nan;
            
            % pairwise active centroid of these two time points
            centroidDist = squareform(pdist(temp));
            [ix,iy] = find(centroidDist > centroid_sep(bin) & centroidDist <= centroid_sep(bin+1));
            firstCen = [firstCen;ds(ix)];
            secondCen = [secondCen;ds(iy)];
        end
        % merge
        shiftCentroid{bin,rp} = firstCen.*secondCen;
        C = corrcoef(firstCen,secondCen);  % correlation
        aveShiftRW(bin,rp) = C(1,2);
    end
end

allMeanCorr = nanmean(aveShiftRW,2)';
% allMeanCov(sd,:) = nanmean(allCov,2)';

figure
hold on
plot(aveShiftRW,'LineWidth',1.5,'Color',greys(6,:))
plot(allMeanCorr,'LineWidth',3,'Color',blues(8,:))
hold off
box on
xlabel('Distance (bin)')
ylabel('$\rho$','Interpreter','latex')

% ************************************************************************
% plot the experiment and random walk in the same figure
% ************************************************************************
figure
hold on
fh = shadedErrorBar((1:size(aveShift,1))',aveShift',{@mean,@std});
box on
set(fh.edge,'Visible','off')
fh.mainLine.LineWidth = 4;
fh.mainLine.Color = blues(10,:);
fh.patch.FaceColor = blues(7,:);

fh2 = shadedErrorBar((1:size(aveShiftRW,1))',aveShiftRW',{@mean,@std});
box on
set(fh2.edge,'Visible','off')
fh2.mainLine.LineWidth = 3;
fh2.mainLine.Color = greys(7,:);
fh2.patch.FaceColor = greys(5,:);
hold off

legend('Experiment','Random walk')
set(gca,'XTick',0:2:6,'XTickLabel',{'0','0.2','0.4','0.6'})
% title(['$\Delta t = ',num2str(deltaTau),'$'],'Interpreter','latex')
xlabel('Distance (L)')
ylabel('$\rho$','Interpreter','latex')

% figPref = [sFolder,filesep,'hipp_centroid_shift_exp_rw'];
% saveas(gcf,[figPref,'.fig'])
% print('-depsc',[figPref,'.eps'])


% load the 1D place cell model to make direct comparision
modelFile = '../data/1D_slice_centroidCorr_0708.mat';
pc1Dmodel = load(modelFile,'aveShift','aveShiftM');

% ********************************************************************************
% compare model and experiment data together
% ********************************************************************************
barFig = figure;
pos(3)=3.5;  
pos(4)=2.8;
set(gcf,'color','w','Units','inches','Position',pos)

hold on
fh = errorbar((1:size(aveShift,1))',nanmean(aveShift,2),nanstd(aveShift,0,2),'o-',...
    'MarkerFaceColor',blues(9,:),'MarkerEdgeColor',blues(9,:),'LineWidth',1.5,...
    'Color',blues(9,:),'MarkerSize',6,'CapSize',0);


fh3 = shadedErrorBar(distR',selModelData',{@mean,@std});
box on
set(fh3.edge,'Visible','off')
fh3.mainLine.LineWidth = 1.5;
fh3.mainLine.Color = reds(10,:);
fh3.patch.FaceColor = reds(7,:);

fh2 = shadedErrorBar((1:size(aveShiftRW,1))',aveShiftRW',{@mean,@std});
box on
set(fh2.edge,'Visible','off')
fh2.mainLine.LineWidth = 1.5;
fh2.mainLine.Color = greys(7,:);
fh2.patch.FaceColor = greys(5,:);
hold off

box on
hold off
xlabel('Distance (L)','FontSize',20)
ylabel('$\rho$','Interpreter','latex','FontSize',20)
set(gca,'FontSize',16,'LineWidth',1)
lg = legend('Experiment','Model','Random walk');
set(gca,'XTick',0:2:6,'XTickLabel',{'0','0.2','0.4','0.6'})


% figPref = [sFolder,filesep,'hipp_centroid_shift_exp_rw_model_1Dslice_errorbar'];
% saveas(gcf,[figPref,'.fig'])
% print('-depsc',[figPref,'.eps'])


%% Publication-ready figures

sFolder = '/Users/shawnqin/OneDrive - Harvard University/MATLAB/representationDrift/figures';
figWidth = 3.5;      % width of figure
figHeight = 2.8;   % height of figure


% ****************************************************
% Fraction of active neurons, 5 day concantenated
% ****************************************************
barFig5Day = figure;
pos(3)=figWidth;  
pos(4)=figHeight;
set(gcf,'color','w','Units','inches','Position',pos)

% bh = bar(actiFracSess,0.7,'FaceColor',blues(5,:),'EdgeColor',greys(7,:),'LineWidth',1.5);
bh = bar(actFrac,0.7,'EdgeColor',greys(7,:),'LineWidth',1.5);
bh.FaceColor = 'flat';
for i = 1:5
    bh.CData(i,:) = blues(1+2*i,:);
end
ylim([0,0.3])
set(gca,'xticklabel',{'5','10','15','20','25'},'FontSize',16)
% xtickangle(40)
xlabel({'days'},'FontSize',20)
ylabel({'fraction of','place cells'},'FontSize',20)

%save the figures
% figPref = [sFolder,filesep,'hipp_frac_acti_5day'];
% saveas(barFig5Day,[figPref,'.fig'])
% print('-depsc',[figPref,'.eps'])

% ******************************************
% population vector if order based on day 1
% ******************************************
figPVcorr = figure;
hold on
% set(gcf,'renderer','Painters')
pos(3)=figWidth;  
pos(4)=figHeight;
set(figPVcorr,'color','w','Units','inches','Position',pos)

daysep = [1:5,16:20];
errorbar(daysep(1:5)',PVcorr(1:5,1),PVcorr(1:5,2),'o-','MarkerSize',errSymSize,'MarkerFaceColor',blues(9,:),...
    'MarkerEdgeColor',blues(9,:),'Color',blues(9,:),'LineWidth',1.5,'CapSize',0)
errorbar(daysep(6:10)',PVcorr(6:10,1),PVcorr(6:10,2),'o-','MarkerSize',errSymSize,'MarkerFaceColor',blues(9,:),...
    'MarkerEdgeColor',blues(9,:),'Color',blues(9,:),'LineWidth',1.5,'CapSize',0)
xlabel('Days','FontSize',labelFontSize)
ylabel('PV correlation','FontSize',labelFontSize)
box on
% xlim([1,5])
ylim([0.1,1])
set(gca,'FontSize',axisFont,'LineWidth',axisLw,'XTick',[0:2:4,16:2:20])
% set(gca,'XTick',[0:2:4,16:2:20])

breakxaxis([6,15])

%save the figures
% figPref = [sFolder,filesep,'hipp_data_pvCorr'];
% saveas(figPVcorr,[figPref,'.fig'])
% print('-depsc',[figPref,'.eps'])


% ******************************************
% Shift of centroids
% ******************************************
figShift = figure;
hold on
% set(gcf,'renderer','Painters')
pos(3)=figWidth;  
pos(4)=figHeight;
set(figShift,'color','w','Units','inches','Position',pos)

daysSel = [1,10,50];
for i = 1:length(daysSel)
     errorbar(shiftMean(daysSel(i),:)',shiftStd(daysSel(i),:)'./size(centroidShift_P,1),'-o',...
        'MarkerSize',4,'Color',blues(1+3*i,:),'MarkerEdgeColor',blues(1+3*i,:),...
        'MarkerFaceColor',blues(1+3*i,:),'CapSize',0,'LineWidth',1.5)      
end

% add randomly shuffled
for i = 1:length(daysSel)
    errorbar(shiftMeanR(daysSel(i),:)',shiftStdR(daysSel(i),:)'./size(centroidShift_P_R,1),'-',...
        'MarkerSize',4,'Color',oranges(1+3*i,:),'MarkerEdgeColor',oranges(1+3*i,:),...
        'MarkerFaceColor',oranges(1+3*i,:),'CapSize',0,'LineWidth',1.5)
end
box on
% legend('1 day','5 days','10 days','30 days','50 days')
lg = legend('1 day','10 days','50 days');
set(lg,'FontSize',14)

ylim([0 0.15])
xlim([1 binWidth])
xticks(5:10:binWidth+1)
x = (-binWidth/2 + 5):10:binWidth/2;
xticklabels({x});
xlabel('Centroid shift','FontSize',labelFontSize)
ylabel('Fraction','FontSize',labelFontSize)
set(gca,'FontSize',axisFont)

%save the figures
% figPref = [sFolder,filesep,'hipp_data_shift'];
% saveas(figShift,[figPref,'.fig'])
% print('-depsc',[figPref,'.eps'])

%% 
% ****************************************
% Population vector and similarity matrix
% ****************************************
fh_rep1 = figure;
pos(3)=3;  
pos(4)=3;
set(fh_rep1,'color','w','Units','inches','Position',pos)

imagesc(Y1)
set(gca,'XTick','','YTick','','LineWidth',0.5,'Visible','off')

% figPref = [sFolder,filesep,'hipp_repr_day1'];
% saveas(fh_rep1,[figPref,'.fig'])
% print('-depsc',[figPref,'.eps'])


fh_rep2 = figure;
pos(3)=3;  
pos(4)=3;
set(fh_rep2,'color','w','Units','inches','Position',pos)

imagesc(Y2)
set(gca,'XTick','','YTick','','LineWidth',0.5,'Visible','off')

% figPref = [sFolder,filesep,'hipp_repr_day16'];
% saveas(fh_rep2,[figPref,'.fig'])
% print('-depsc',[figPref,'.eps'])


% simiarity matrix
fh_sm1 = figure;
pos(3)=3;  
pos(4)=3;
set(fh_sm1,'color','w','Units','inches','Position',pos)

imagesc(Y1'*Y1,[0,80])
set(gca,'XTick','','YTick','','LineWidth',0.5,'Visible','off')
% 
% figPref = [sFolder,filesep,'hipp_SM1_day1'];
% saveas(fh_sm1,[figPref,'.fig'])
% print('-depsc',[figPref,'.eps'])
% 
fh_sm2 = figure;
pos(3)=3;  
pos(4)=3;
set(fh_sm2,'color','w','Units','inches','Position',pos)

imagesc(Y2'*Y2,[0,80])
set(gca,'XTick','','YTick','','LineWidth',0.5,'Visible','off')

% figPref = [sFolder,filesep,'hipp_SM2_day16'];
% saveas(fh_sm2,[figPref,'.fig'])
% print('-depsc',[figPref,'.eps'])
% 