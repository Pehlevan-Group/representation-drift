% Detailed drift analysis of 2D place cells
% Peak position is based on the center of mass, previously we used peak
% positions

close all
clear

%% Select data to analyze
% dFile = './data/pc2D_Batch_Np_400std0.01_l1_0.17_l2_0.01_step1_lr0.05.mat';
dFile = './data/batch_new_N/pc2D_Batch_Np_400std0.01_l1_0.17_l2_0.01_step1_lr0.05.mat';
load(dFile);

%% Graphics setting
% this part polish some of the figures and make them publication ready
% define all the colors
sFolder = './figures';
figPre = 'placeCell2D_';

nc = 256;   % number of colors
spectralMap = flip(brewermap(nc,'Spectral'));
PRGnlMap = brewermap(nc,'PRGn');
% RdBuMap = flip(brewermap(nc,'RdBu'),1);

blues = brewermap(11,'Blues');
BuGns = brewermap(11,'BuGn');
greens = brewermap(11,'Greens');
greys = brewermap(11,'Greys');
PuRd = brewermap(11,'PuRd');
YlOrRd = brewermap(11,'YlOrRd');
oranges = brewermap(11,'Oranges');
RdBuMap = flip(brewermap(nc,'RdBu'),1);
BlueMap = flip(brewermap(nc,'Blues'),1);
GreyMap = brewermap(nc,'Greys');


figWidth = 3.2;
figHeight = 2.8;
lineWd = 1.5;
symbSize = 4;
labelSize = 20;
axisSize = 16;
axisWd = 1;

% figure size, weight and height
pos(3)=figWidth;  
pos(4)=figHeight;


%%
% find the time point of re-appear based on peakFlags

allEmergeDist = [];
aveEmergeDist = nan(size(pkAmp,1),1);
actiShift = [];  % store the shif of active period
inacInter = [];  % store interval of silent period
actiInter = [];  % store interval of active period
appAmp = [];

neuronIndicesActi = [];   % store the neuron index for active periods
neuronIndicesSilen = [];  % store the neuron index for silent periods
instShift = cell(param.Np,1);

% x and y position of center of mass
X_center = squeeze(pkCenterMass(:,1,:));
Y_center = squeeze(pkCenterMass(:,2,:));

pkFlags = ~isnan(pks);
for i = 1:size(pkAmp,1)
    temp = diff(pkFlags(i,:));
    disappearInx = find(temp==-1);
    reappearInx = find(temp ==1)+1; % index of reappear
    
    disPosi = pks(i,disappearInx);
    appPosi = pks(i,reappearInx);
    
    if ~isempty(reappearInx)
        if reappearInx(1) > disappearInx(1)
            x_app = X_center(i,reappearInx);
            y_app = Y_center(i,reappearInx);
            
            x_disap = X_center(i,disappearInx(1:length(reappearInx)));
            y_disap = Y_center(i,disappearInx(1:length(reappearInx)));
            
            % time interval of silent period
            timeInter = reappearInx - disappearInx(1:length(reappearInx));
            appAmp = [appAmp,pkAmp(i,reappearInx)];
            
            % time interval of active period
            timeInterActi = disappearInx(2:length(x_disap)) - reappearInx(1:length(x_disap)-1);
            t1_acti = abs(x_disap(2:end) - x_app(1:end-1));
            t2_acti = abs(y_disap(2:end) - y_app(1:end-1));
        else
            numEvents = length(appPosi)-1;
            x_app = X_center(i,reappearInx(2:end));
            y_app = Y_center(i,reappearInx(2:end));
            
            x_disap = X_center(i,disappearInx(1:numEvents));
            y_disap = Y_center(i,disappearInx(1:numEvents));
            
            timeInter = reappearInx(2:end) - disappearInx(1:numEvents);
            appAmp = [appAmp,pkAmp(i,reappearInx(2:end))];
            
            % time interval of active period
            timeInterActi = disappearInx(1:numEvents) - reappearInx(1:numEvents);
            t1_acti = abs(x_disap - x_app);
            t2_acti = abs(y_disap - y_app);
        end


        t1 = abs(x_app - x_disap);
        t2 = abs(y_app - y_disap);

        dx = min(t1,param.ps - t1);
        dy = min(t2,param.ps - t2);
        allEmergeDist = [allEmergeDist,sqrt(dx.^2 + dy.^2)];
        aveEmergeDist(i) = mean(sqrt(dx.^2 + dy.^2));
        
        % active period start and end

        % periodic boundary conditions
        dx_acti = min(t1_acti,param.ps - t1_acti);
        dy_acti = min(t2_acti,param.ps - t2_acti);
        actiShift = [actiShift,sqrt(dx_acti.^2 + dy_acti.^2)];
        
        inacInter = [inacInter,timeInter];
        actiInter = [actiInter, timeInterActi];
        
        % Store the neuron index
        neuronIndicesActi = [neuronIndicesActi,ones(1,length(timeInterActi))*i]; % provides the 
        neuronIndicesSilen = [neuronIndicesSilen,ones(1,length(timeInter))*i];
        
        % instantaneous amplitude and stability
        Ia = find(pkFlags(i,:));
        temp = pkAmp(i,Ia);
        xPosi = X_center(i,Ia);
        yPosi = Y_center(i,Ia);
        dxs = abs(diff(xPosi));
        dys = abs(diff(yPosi));
        dxs = min(dxs,param.ps - dxs);
        dys = min(dys,param.ps - dys);
        allShift = sqrt(dxs.^2 + dys.^2);
        noJump = find(diff(Ia)==1);
        instShift{i} = [allShift(noJump)',temp(noJump)'];
        
        
%         activeInter = [activeInter,temp(temp>1)-1];
        
    end
end


figure
plot(inacInter(:),allEmergeDist(:),'o')


% correlation between jump shift and inactive period interval
f_jumpAmp= figure;
set(f_jumpAmp,'color','w','Units','inches','Position',pos)

plot(inacInter(:),allEmergeDist(:),'o','MarkerSize',symbSize,...
    'MarkerEdgeColor',greys(9,:),'LineWidth',lineWd)
xlabel('inactive interval','FontSize',labelSize)
ylabel('$\Delta r$','Interpreter','latex','FontSize',labelSize)
set(gca,'FontSize',axisSize,'LineWidth',axisWd)


% histogram of jump distance
f_jumpHis= figure;
set(f_jumpHis,'color','w','Units','inches','Position',pos)

histogram(allEmergeDist)
xlabel('$\Delta r$','Interpreter','latex','FontSize',labelSize)
ylabel('Count','FontSize',labelSize)
set(gca,'FontSize',axisSize,'LineWidth',axisWd)


% Average shift during the active period
f_actiShift= figure;
set(f_actiShift,'color','w','Units','inches','Position',pos)

histogram(actiShift)
xlabel('$\Delta r$','Interpreter','latex','FontSize',labelSize)
ylabel('Count','FontSize',labelSize)
set(gca,'FontSize',axisSize,'LineWidth',axisWd)


% correlation between active time interval and relative shift
actiTime_shift_h = figure;
set(actiTime_shift_h,'color','w','Units','inches','Position',pos)

plot(actiInter(:),actiShift(:),'o','MarkerSize',symbSize,...
    'MarkerEdgeColor',greys(9,:),'LineWidth',lineWd)
xlabel('active interval','FontSize',labelSize)
ylabel('$\Delta r$','Interpreter','latex','FontSize',labelSize)
set(gca,'FontSize',axisSize,'LineWidth',axisWd)

%% 
% ************************************************
% Analysis of individual neurons
% ************************************************

% estimate linear relationship by regression
% silent periods
allRsqSilen = [];
allRsqActi = [];
signiSilen =[];
signiActi = [];
for i = 1:size(pkFlags,1)
    indx = neuronIndicesSilen == i;
    y = allEmergeDist(indx)';
    if ~isempty(y) && length(y)>2
        [C, P] = corrcoef(inacInter(indx)',y);
        allRsqSilen = [allRsqSilen,C(1,2)];
        signiSilen = [signiSilen,P(1,2)];
    end
    
    % active period shift vs length active period
    inx_a = neuronIndicesActi==i;
    ya = actiShift(inx_a)';
    if ~isempty(ya) && length(ya)>2
        [C, P] = corrcoef(actiInter(inx_a)',ya);
        allRsqActi = [allRsqActi,C(1,2)];
        signiActi = [signiActi,P(1,2)];
    end
    
    
end

% Fit goodness, R-square
figure
hold on
histogram(allRsqActi,20)
histogram(allRsqSilen,20)
hold off
legend('active','silent')
box on
xlabel('$\rho$','Interpreter','latex')
ylabel('Count')

% Significance
figure
hold on
plot(allRsqActi,signiActi,'^')
plot(allRsqSilen,signiSilen,'o')
hold off
box on
legend('Active','Silent')
xlabel('$\rho$','Interpreter','latex')
ylabel('P value')
set(gca,'YScale','log','YLim',[1e-3,1])


% dr and silent interval, randomly select 5 neurons
neuronInx =  randperm(size(pkFlags,1),5);
symbolList = {'o','+','d','^','<'};
colors = brewermap(5,'Set1');

silent_shift = figure;
set(silent_shift,'color','w','Units','inches','Position',pos)

hold on
for i = 1:length(neuronInx)
    indices = neuronIndicesSilen == neuronInx(i);
    plot(inacInter(indices),allEmergeDist(indices),symbolList{i},'MarkerSize',symbSize,...
    'MarkerEdgeColor',colors(i,:),'LineWidth',lineWd)
end
hold off
xlabel('inactive interval','FontSize',labelSize)
ylabel('$\Delta r$','Interpreter','latex','FontSize',labelSize)
set(gca,'FontSize',axisSize,'LineWidth',axisWd)
box on


% dr during active interval
acti_shift = figure;
set(acti_shift,'color','w','Units','inches','Position',pos)

hold on
for i = 1:length(neuronInx)
    indices = neuronIndicesSilen == neuronInx(i);
    plot(actiInter(indices),actiShift(indices),symbolList{i},'MarkerSize',symbSize,...
    'MarkerEdgeColor',colors(i,:),'LineWidth',lineWd)
end
hold off
xlabel('Active time interval','FontSize',labelSize)
ylabel('$\Delta r$','Interpreter','latex','FontSize',labelSize)
set(gca,'FontSize',axisSize,'LineWidth',axisWd)
box on


%% Instantaneous shift
neuronSel = randperm(param.Np,5);  % randomly select 5 neurons

figure
hold on
for i = 1:5
    temp = instShift{neuronSel(i)}(:,1) >=1;
    if ~isempty(temp)
        plot(instShift{neuronSel(i)}(temp,2),instShift{neuronSel(i)}(temp,1),'.','Color','k')
    end
end
hold off
box on
xlabel('Amplitude')
ylabel('$\Delta r_t$','Interpreter','latex')


% using all the data and concantenate the data
drts = [];   % store all the dr
ampts = [];  % store all the amplitude

figure
hold on
for i = 1:param.Np
    if ~isempty(instShift{i})
        temp = instShift{i}(:,1) >=1;
        if ~isempty(temp)
            plot(instShift{i}(temp,2),instShift{i}(temp,1),'.','Color','k')
            drts = [drts;instShift{i}(temp,1)];
            ampts = [ampts;instShift{i}(temp,2)];
        end
    end
end
hold off
box on
xlabel('Amplitude')
ylabel('$\Delta r_t$','Interpreter','latex')

% bin the data
ampCenters = [0:1:8,12];
meanDrs = nan(length(ampCenters)-1,2);
ampAveStd = nan(length(ampCenters)-1,2);
for i = 1:length(ampCenters)-1
    inx = ampts > ampCenters(i) & ampts <= ampCenters(i+1);
    meanDrs(i,:) = [mean(drts(inx)),std(drts(inx))];
    ampAveStd(i,:) = [mean(ampts(inx)),std(ampts(inx))];
end

figure
errorbar(ampAveStd(:,1),meanDrs(:,1),-meanDrs(:,2),meanDrs(:,2),-ampAveStd(:,2),...
    ampAveStd(:,2),'o','MarkerSize',symbSize,'MarkerEdgeColor',blues(9,:),'LineWidth',lineWd)
xlabel('Amplitdue')
ylabel('$\Delta r_t$','Interpreter','latex')
