% Use CMA-ES to search for the best fitted distribution of dS
% This is used to generate data for Fig 5J
% Due to the stochastic nature of the optimization, you might need to run
% the simulation a couple of times to get the best solution

% load the data, specify the directory of data
load('../data/pc_1dayShift_exper.mat',...
    'allCentroids','mergeShifts')
% set the parameters

% *****************************************************************************
% add the opitmization program file folder to the path, we used external
% CMA-ES algorithm: http://www.cmap.polytechnique.fr/~nikolaus.hansen/cmaes_inmatlab.html
addpath '/Users/shawnqin/Documents/MATLAB/olfaction'
%*****************************************************************************

global param targetData oneDayShift x_eval targetECDF
param.L = 50;  % length of the linear track
% param.rho = mean(mergeShifts>0);  % fraction of centroid shifts

targetData = allCentroids(allCentroids>0);  % centroid position
oneDayShift = mergeShifts;   % one day shifts

% select fitted distribution
fitDist = 'LevyAlp';   % or 'beta'
% initialization of the optimizer
opts = cmaes;
iniSig = 2;               % the inial sigma along each dimension
opts.MaxIter = 2e4;       % maximum number of iteration
opts.LBounds = 0;         % lower bounds, default
opts.UBounds = 10;        % default, not specfied
opts.PopSize = 5;

if strcmp(fitDist,'exp')
    w0 = rand;
    [wmin,fmin,~,~,~,bestever] = cmaes('fitExp',w0,iniSig,opts);
elseif strcmp(fitDist,'beta')
    w0 = rand(1,2);
    [wmin,fmin,~,~,~,bestever] = cmaes('fitBeta',w0,iniSig,opts);
elseif strcmp(fitDist,'gauss')
    w0 =rand;
    [wmin,fmin,~,~,~,bestever] = cmaes('fitGauss',w0,iniSig,opts);
elseif strcmp(fitDist,'LevyAlp')
    opts.LBounds = [0.001;0.001];       % lower bounds, default
    opts.UBounds = [1.5;0.99];       % default, not specfied
    w0 =rand(1,2);
%     w0 = [1,0.1];
    [wmin,fmin,~,~,~,bestever] = cmaes('fitLevyAlpStable',w0,0.2,opts);
elseif strcmp(fitDist,'LevyAlpEDF')
    % fit based on the ecdf of alpha stable distribution
    [f, x] = ecdf(oneDayShift/param.L);
    x_eval = min(x):0.005:max(x);
    for i = 1:length(x_eval)
        targetECDF(i) = max(f(x<=x_eval(i)));
    end
    
    opts.LBounds = [0.001;0.001];       % lower bounds, default
    opts.UBounds = [1.5;0.99];       % default, not specfied
    w0 =rand(1,2);
    [wmin,fmin,~,~,~,bestever] = cmaes('fitLevyAlpStable_nlsq',w0,0.2,opts);
end


%% plot and compare the histogram of dr with experiments

% generate the 1-step difference based on the fitted distribution
N  = length(targetData);
ds =  stblrnd(wmin(1),0,wmin(2),0,N,1);
ds(ds>1)=1;
ds(ds<-1) = -1;
rRaw = ds.*param.L + targetData;  % new position

% refractory boundary condition
inx1 = rRaw > param.L;
rRaw(inx1) = max(2*param.L - rRaw(inx1),0);
inx2 = rRaw < 0;
rRaw(inx2) = min(abs(rRaw(inx2)),param.L);

% shfit of centroid
dr = rRaw - targetData;

% first define some colors
rdBu = brewermap(11,'RdBu');
greys = brewermap(11,'Greys');

figure
hold on
histogram(dr'/param.L,40,'Normalization','pdf')
histogram(oneDayShift'/param.L,40,'Normalization','pdf')
hold off
box on
xlim([-1,1])
xlabel('$\Delta r$','Interpreter','latex')
ylabel('pdf')
legend('model','experiment')

% A slight different way to represent the results
edges = -1:0.05:1;
nM = histcounts(dr'/param.L,edges);
nE = histcounts(oneDayShift'/param.L,edges);

histFig = figure;
hold on
pos(3)=3.3;  
pos(4)=2.5;
set(gcf,'color','w','Units','inches','Position',pos)
plot(edges(2:end),nM/sum(nM),'o-','Color',greys(8,:),'MarkerSize',4,'MarkerEdgeColor',...
    greys(8,:),'MarkerFaceColor',greys(8,:),'LineWidth',1.5)
plot(edges(2:end),nE/sum(nE),'o-','Color',rdBu(10,:),'MarkerSize',4,'MarkerEdgeColor',...
    rdBu(10,:),'MarkerFaceColor',rdBu(10,:),'LineWidth',1.5)
hold off
box on
xlim([-1,1])
ylim([0,0.35])
xlabel('$\Delta r$','Interpreter','latex')
ylabel('Probability')
lg = legend('random walk','experiment');
set(lg,'FontSize',14)
set(gca,'FontSize',16)

% sFolder = '../figures';
% figPref = [sFolder,filesep,'hipp_centroid_shift_LevyAlpStable_rw_model_stepSize_06122022'];
% saveas(gcf,[figPref,'.fig'])
% print('-depsc',[figPref,'.eps'])


% distribution of model step size
sizeFig = figure;
hold on
pos(3)=3;  
pos(4)=2.5;
set(sizeFig,'color','w','Units','inches','Position',pos)
x = -1:0.01:1;
% y = exppdf(abs(x),wmin);
y = stblpdf(x,wmin(1),0,wmin(2),0,'quick');
% y = stblpdf(x,1,0,0.06,0,'quick');

plot(x,y,'LineWidth',3,'Color',greys(8,:))
hold off
box on
xlabel('$\Delta s$','Interpreter','latex')
ylabel('Pdf')
set(gca,'FontSize',16)
% 
% figPref = [sFolder,filesep,'hipp_shift_LevyStable_model_stepSize_06122022'];
% saveas(gcf,[figPref,'.fig'])
% print('-depsc',[figPref,'.eps'])


