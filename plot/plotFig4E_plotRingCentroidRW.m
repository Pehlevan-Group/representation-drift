% compare the centroid position of the ring model
% show that centroid distribution in the model is more "uniform" than the
% independent random walk senario

close all
clear

%% Graphics settings
sFolder = '../figures';
figPre = 'placeRing_Centroid_RW';

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


%% Compare with independent random walk

dFolder = '../data/pcRing_centerRW/';
allFile = dir([dFolder,filesep,'*.mat']);
files = {allFile.name}';

s1 = '(?<= *_Np)[\d]+(?=_)';

numNs = 10;    % could be 9 or 10, depending on the dataset
actiNs = nan(numNs,1);   % store the effective Ns in the interacting model
Ns = nan(length(files),1);  % the Np in simulation
varDs = nan(length(files),2);   % store all the mean variance and std of variance

% first, concantenate N > 
for i0 = 1:length(files)
    Ns(i0) = str2num(char(regexp(files{i0},s1,'match')));
    raw = load(char(fullfile(dFolder,filesep,files{i0})));
    
    if raw.Np > 1
        meanVarDs = nan(raw.repeats,1);
        for k = 1:raw.repeats
            meanVarDs(k) = mean(raw.allCentroidDist{k}(:,2));  
        end
        varDs(i0,:) = [mean(meanVarDs),std(meanVarDs)];
        actiNs(i0) = mean(raw.allAveActi);
    end
end

% sort Ns
[~,ix] = sort(Ns);

plot(varDs(ix))

% sorted variance of centroids
sortedVarDsInteract = varDs(ix(2:end),:);
effNs_sorted = round(actiNs(ix(2:end)));

%% Mimic indpendent random walk, bootstrap from single neuron
raw = load(char(fullfile(dFolder,filesep,files{ix(1)})));
allIndependentDs = nan(numNs-1,2);
for i0 = 1:numNs-1
    % randomly select df
    varDs_eachN = nan(40,1);
    for j = 1:40
        trajSel = randperm(100,effNs_sorted(i0));
        mergeTraj = cat(1,raw.allPks{trajSel});
        
        % estimate the centroid distance
        nearestDist = nan(size(mergeTraj,2),2);  % store the mean and std
        for k = 1:size(mergeTraj,2)
            temp = sort(mergeTraj(~isnan(mergeTraj(:,k)),k));
            ds = [diff(temp);(temp(1)-temp(end)) + 2*pi];
            nearestDist(k,:) = [mean(ds),var(ds)];
        end
        varDs_eachN(j) = mean(nearestDist(:,2));
    end
    
    allIndependentDs(i0,:) = [mean(varDs_eachN),std(varDs_eachN)];
end


%% Plot the figure

f_centerComp = figure;
set(f_centerComp,'color','w','Units','inches','Position',pos)
hold on
errorbar(effNs_sorted,allIndependentDs(:,1),allIndependentDs(:,2),'o-','MarkerSize',...
    symbSize,'MarkerFaceColor',greys(9,:),'MarkerEdgeColor',greys(9,:),...
    'Color',greys(9,:),'LineWidth',lineWd,'CapSize',0)
errorbar(effNs_sorted,sortedVarDsInteract(:,1),sortedVarDsInteract(:,2),'o-','MarkerSize',...
    symbSize,'MarkerFaceColor',blues(9,:),'MarkerEdgeColor',blues(9,:),...
    'Color',blues(9,:),'LineWidth',lineWd,'CapSize',0)
box on
legend('independent','model')
xlabel('$N_{\rm{active}}$','Interpreter','latex','FontSize',20)
ylabel('$\langle(\Delta s - \bar{\Delta s})^2 \rangle (\rm{rad}^2)$','Interpreter','latex','FontSize',20)
% ylabel('$\langle(\Delta s - \langle\Delta s\rangle)^2 \rangle (\rm{rad}^2)$','Interpreter','latex','FontSize',20)
% set(gca,'YScale','linear','XScale','linear','FontSize',16)
set(gca,'YScale','log','XScale','linear','YTick',10.^(-2:1:1),'FontSize',16)
% set(gca,'YScale','log','XScale','linear','FontSize',16)

xlim([0,30])

% prefix = [figPre, 'comparison_log'];
% saveas(f_centerComp,[sFolder,filesep,prefix,'.fig'])
% print('-depsc',[sFolder,filesep,prefix,'.eps'])


% **************************************************
% Linear y axis
% **************************************************
f_centerComp = figure;
set(f_centerComp,'color','w','Units','inches','Position',pos)
hold on
errorbar(effNs_sorted(2:end-1),allIndependentDs(2:end-1,1),allIndependentDs(2:end-1,2),'o-','MarkerSize',...
    symbSize,'MarkerFaceColor',greys(9,:),'MarkerEdgeColor',greys(9,:),...
    'Color',greys(9,:),'LineWidth',lineWd,'CapSize',0)
errorbar(effNs_sorted(2:end-1),sortedVarDsInteract(2:end-1,1),sortedVarDsInteract(2:end-1,2),'o-','MarkerSize',...
    symbSize,'MarkerFaceColor',blues(9,:),'MarkerEdgeColor',blues(9,:),...
    'Color',blues(9,:),'LineWidth',lineWd,'CapSize',0)
box on
legend('independent','model')
xlabel('$N_{\rm{active}}$','Interpreter','latex','FontSize',20)
ylabel('$\langle(\Delta s - \bar{\Delta s})^2 \rangle (\rm{rad}^2)$','Interpreter','latex','FontSize',20)
% ylabel('$\langle(\Delta s - \langle\Delta s\rangle)^2 \rangle (\rm{rad}^2)$','Interpreter','latex','FontSize',20)
% set(gca,'YScale','linear','XScale','linear','FontSize',16)
% set(gca,'YScale','linear','XScale','linear','YTick',10.^(-2:1:1),'FontSize',16)
set(gca,'YScale','linear','XScale','linear','FontSize',16)

% prefix = [figPre, 'comparison_linear'];
% saveas(f_centerComp,[sFolder,filesep,prefix,'.fig'])
% print('-depsc',[sFolder,filesep,prefix,'.eps'])