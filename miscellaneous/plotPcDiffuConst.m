% Plot diffusion constants of place cells in 2D environment
% 

close all
clear

% graphics settings
defaultGraphicsSetttings
greys = brewermap(11,'Greys');
blues = brewermap(11,'blues');

saveFolder = './figures';
figWidth = 3.5;
figHeight = 2.8;

%% load and prepare the data
% Dependence of neural number
dFolder = './data/pcDiffuCont';
% dFolder = './data/pc1D_diff_N';
% dFolder = './data/pc2D_batch_N';

allFile = dir([dFolder,filesep,'*.mat']);
files = {allFile.name}';

aveDs = nan(length(files),1);  % store the average diffusion constants
stdDs = nan(length(files),1);

rfMass = nan(length(files),2);  % mean and standard deviation of RF mass
NpExtract = nan(length(files),1);
actiFrac = nan(length(files),1);   % store the fraction of active neurons
aveAmp = nan(length(files),2);  % store the average firing of
meanNearDist = nan(length(files),2);  %  nearest neighbor distance
for i0 = 1:length(files)
    if isempty(files)
        error('folder is empty!')
    else   
        s1 = '(?<= *Np_)[\d]+(?=.)';
        Np = str2num(char(regexp(files{i0},s1,'match')));
        if ~isempty(Np)
            NpExtract(i0) = Np;
            temp = load(char(fullfile(dFolder,filesep,files{i0})));
            aveDs(i0) = nanmean(temp.Ds);
            stdDs(i0) = nanstd(temp.Ds);
            
            rawMass = temp.pkMas(:);
            rfMass(i0,:) = [nanmean(rawMass(rawMass < 200)),nanstd(rawMass(rawMass < 200))];
            
            % fraction of active neurons
            actiFrac(i0) = mean(sum(~isnan(temp.pkAmp),1))/Np;
            aveAmp(i0,:) = [nanmean(temp.pkAmp(:)),nanstd(temp.pkAmp(:))];
            
            % average nearest neighbor
            nearestDist = nan(size(temp.pkCenterMass,3),1);
            for j = 1:size(temp.pkCenterMass,3)
                cmPosi = temp.pkCenterMass(:,:,j);
                pd = pdist(cmPosi(~isnan(cmPosi(:,1)),:));
                euclDistM = squareform(pd);   % distance matrix
                sortedDist = sort(euclDistM,2,'ascend');
                nearestDist(j) = mean(sortedDist(:,2)); % the frist column are zeros  
            end
            meanNearDist(i0,:) = [mean(nearestDist),std(nearestDist)];
        end
    end
end
%% if there are small N needed to add

% small Nps
sfInx = find(isnan(NpExtract));    % index of file storing the small Np simulations
smallNpFile = fullfile(dFolder,filesep,files{sfInx});
% smallNpFile = './data/pcDiffSmallNp_1D.mat';

temp = load(smallNpFile);

allDs = temp.allDs;
smNps = nan(length(allDs),1);
aveDsmNp = nan(length(allDs),1);
stdDsmNp = nan(length(allDs),1);
for i = 1:length(smNps)
    smNps(i) = size(allDs{i},1);
    aveDsmNp(i) = nanmean(allDs{i}(:));
    stdDsmNp(i) = nanstd(allDs{i}(:));
end

% merge the data togehter
[~, sortedInx] = sort(NpExtract);
aveDsAll = [aveDsmNp;aveDs(sortedInx(1:end-1))];
stdDsAll = [stdDsmNp;stdDs(sortedInx(1:end-1))];
allNps = [smNps;NpExtract(sortedInx(1:end-1))];


% plot N dependent noise diffusion constants

NDfig = figure;
pos(3)=figWidth; 
pos(4)=figHeight;
set(NDfig,'color','w','Units','inches','Position',pos)
hold on

errorbar(allNps,aveDsAll,stdDsAll,'o-','MarkerSize',6,'Color',greys(10,:),...
    'MarkerEdgeColor',greys(10,:),'MarkerFaceColor',greys(10,:),'CapSize',0,...
    'LineWidth',1.5)
box on
hold off
xlabel('number of place cells','FontSize',20)
% ylabel('$\langle (\Delta r )^2 \rangle $','Interpreter','latex','FontSize',20)
ylabel('$D$','Interpreter','latex','FontSize',20)
set(gca,'XScale','log','YScale','linear','FontSize',16)

%% N dependent variable plots
% sort the number of N
I = find(~isnan(aveDs));
[Nps, ord] = sort(NpExtract(I),'ascend');

% ********************************************************
% How the effective diffusion costants change with N
% ********************************************************
NDfig = figure;
pos(3)=figWidth; 
pos(4)=figHeight;
set(NDfig,'color','w','Units','inches','Position',pos)
hold on

errorbar(Nps,aveDs(I(ord)),stdDs(I(ord)),'o-','MarkerSize',6,'Color',greys(10,:),...
    'MarkerEdgeColor',greys(10,:),'MarkerFaceColor',greys(10,:),'CapSize',0,...
    'LineWidth',1.5)
box on
hold off
xlabel('number of place cells','FontSize',20)
% ylabel('$\langle (\Delta r )^2 \rangle $','Interpreter','latex','FontSize',20)
ylabel('$D$','Interpreter','latex','FontSize',20)
set(gca,'XScale','linear','YScale','log','FontSize',16,'YTick',10.^(-3:-1))

prefix = 'pc2D_batch_D_N';
saveas(NDfig,[saveFolder,filesep,prefix,'.fig'])
print('-depsc',[saveFolder,filesep,prefix,'.eps'])


% ******************************************
% Active fraction vs the nunber of neurons
% ******************************************
actiN_fig = figure;
pos(3)=figWidth; 
pos(4)=figHeight;
set(actiN_fig,'color','w','Units','inches','Position',pos)

plot(Nps,actiFrac(I(ord)),'o-','MarkerSize',8,'Color',greys(10,:),'MarkerEdgeColor',...
    greys(10,:),'MarkerFaceColor',greys(10,:),'LineWidth',1.5)
xlabel('number of neurons','FontSize',20)
ylabel({'fraction of', 'active time'},'FontSize',20)
set(gca,'XScale','linear','YScale','linear','FontSize',16,'YTick',10.^(-3:-1))

prefix = 'pc2D_batch_acti_N';
saveas(actiN_fig,[saveFolder,filesep,prefix,'.fig'])
print('-depsc',[saveFolder,filesep,prefix,'.eps'])


% ******************************************
% Active fraction vs diffusion contants
% ******************************************
actiN_D_fig = figure;
pos(3)=figWidth; 
pos(4)=figHeight;
set(actiN_D_fig,'color','w','Units','inches','Position',pos)

errorbar(actiFrac(I(ord)),aveDs(I(ord)),stdDs(I(ord)),'o-','MarkerSize',6,'Color',greys(10,:),...
    'MarkerEdgeColor',greys(10,:),'MarkerFaceColor',greys(10,:),'CapSize',0,...
    'LineWidth',1.5)
xlabel('fraction of active time','FontSize',20)
ylabel('$D$','Interpreter','latex','FontSize',20)
set(gca,'XScale','linear','YScale','linear','FontSize',16)

prefix = 'pc2D_batch_acti_D_N';
saveas(actiN_D_fig,[saveFolder,filesep,prefix,'.fig'])
print('-depsc',[saveFolder,filesep,prefix,'.eps'])



% ******************************************
% peak amplitude vs N
% ******************************************
pkAmp_fig = figure;
pos(3)=figWidth; 
pos(4)=figHeight;
set(pkAmp_fig,'color','w','Units','inches','Position',pos)

errorbar(Nps,aveAmp(I(ord),1),aveAmp(I(ord),2),'o-','MarkerSize',6,'Color',greys(10,:),...
    'MarkerEdgeColor',greys(10,:),'MarkerFaceColor',greys(10,:),'CapSize',0,...
    'LineWidth',1.5)
xlabel('Number of neurons','FontSize',20)
ylabel('Peak amplitude','FontSize',20)
set(gca,'XScale','linear','YScale','linear','FontSize',16)

prefix = 'pc2D_batch_Amp_N';
saveas(pkAmp_fig,[saveFolder,filesep,prefix,'.fig'])
print('-depsc',[saveFolder,filesep,prefix,'.eps'])


% ******************************************
% Mean nearest neighbor distance
% ******************************************
actiFrac_fig = figure;
pos(3)=figWidth; 
pos(4)=figHeight;
set(actiFrac_fig,'color','w','Units','inches','Position',pos)

errorbar(Nps,meanNearDist(I(ord),1),meanNearDist(I(ord),2),'o-','MarkerSize',6,'Color',greys(10,:),...
    'MarkerEdgeColor',greys(10,:),'MarkerFaceColor',greys(10,:),'CapSize',0,...
    'LineWidth',1.5)
xlabel('Number of neurons','FontSize',20)
ylabel({'Nearest neighbor','distance'},'FontSize',20)
set(gca,'XScale','linear','YScale','linear','FontSize',16)


% ********************************************
% Mean active fraction
% ********************************************
actiFrac_fig = figure;
pos(3)=figWidth; 
pos(4)=figHeight;
set(actiFrac_fig,'color','w','Units','inches','Position',pos)

plot(Nps,actiFrac(I(ord)),'o-','MarkerSize',6,'Color',greys(10,:),...
    'MarkerEdgeColor',greys(10,:),'MarkerFaceColor',greys(10,:),...
    'LineWidth',1.5)
ylim([0.5,1.05])
xlabel('Number of neurons','FontSize',20)
ylabel({'Fraction of', 'active neuron'},'FontSize',20)
set(gca,'XScale','linear','YScale','linear','FontSize',16)


% ******************************************
% Centroid of example neuron
% ******************************************
sepSel = 5;
numPoints = round(size(pkCenterMass,3)/sepSel);
bluesSpec = flip(brewermap(numPoints,'blues'));
PuRdSpec = flip(brewermap(numPoints,'PuRd'));
sepctrumColors = {bluesSpec,PuRdSpec};

neurSel = randperm(Np,2);

figure
hold on
for i = 1:2
    temp = squeeze(pkCenterMass(neurSel(i),:,:));
    for j = 1:numPoints
        plot(temp(1,sepSel*j),temp(2,sepSel*j),'+','MarkerSize',6,...
    'MarkerEdgeColor',sepctrumColors{i}(j,:),'Color',sepctrumColors{i}(j,:),...
    'LineWidth',1.5)
    end
end
hold off
box on
xlim([0,32])
ylim([0,32])
xlabel('X position')
ylabel('Y position')

%% random walk step dependence

dFolder = './data/pcRWsteps';

allFile = dir([dFolder,filesep,'*.mat']);
files = {allFile.name}';

aveDs = nan(length(files),1);  % store the average diffusion constants
stdDs = nan(length(files),1);

rfMass = nan(length(files),2);  % mean and standard deviation of RF mass
stepExtract = nan(length(files),1);
for i0 = 1:length(files)
    if isempty(files)
        error('folder is empty!')
    else   
        s1 = '(?<= *step)[\d]+(?=_)';
        step = str2num(char(regexp(files{i0},s1,'match')));
        if ~isempty(step)
            stepExtract(i0) = step;
            temp = load(char(fullfile(dFolder,filesep,files{i0})));
            aveDs(i0) = nanmean(temp.Ds);
            stdDs(i0) = nanstd(temp.Ds);
            
            rawMass = temp.pkMas(:);
            rfMass(i0,:) = [nanmean(rawMass(rawMass < 200)),nanstd(rawMass(rawMass < 200))];
        end
    end
end

% plot the figure
figure
% hold on
errorbar(stepExtract,aveDs,stdDs,'o','MarkerSize',10,'Color',greys(10,:),...
    'MarkerEdgeColor',greys(10,:),'MarkerFaceColor',greys(10,:),'CapSize',0,...
    'LineWidth',2)
% plot(stepExtract,intersep*stepExtract.^epns,'Color',greys(8,:))
% hold off
box on
xlabel('random walk speed')
% ylabel('$\langle (\Delta r )^2 \rangle $','Interpreter','latex')
ylabel('$D$','Interpreter','latex')
set(gca,'XScale','linear','YScale','linear')




%% random walk learning rate dependence
dFolder = './data/pcRWlearnRate';
% dFolder = './data/pc1D_diff_lr';

allFile = dir([dFolder,filesep,'*.mat']);
files = {allFile.name}';

aveDs = nan(length(files),1);  % store the average diffusion constants
stdDs = nan(length(files),1);

rfMass = nan(length(files),2);  % mean and standard deviation of RF mass
lrExtract = nan(length(files),1);
for i0 = 1:length(files)
    if isempty(files)
        error('folder is empty!')
    else   
        s1 = '(?<= *lr)[\d\.]+(?=.mat)';
        lr = str2num(char(regexp(files{i0},s1,'match')));
        if ~isempty(lr)
            lrExtract(i0) = lr;
            temp = load(char(fullfile(dFolder,filesep,files{i0})));
            aveDs(i0) = nanmean(temp.Ds);
            stdDs(i0) = nanstd(temp.Ds);
            
            rawMass = temp.pkMas(:);
            rfMass(i0,:) = [nanmean(rawMass(rawMass < 200)),nanstd(rawMass(rawMass < 200))];
        end
    end
end

% log-log fit
logX = [ones(5,1),log(lrExtract(1:end-1))];
logY = log(aveDs(1:end-1));
b = logX\logY;
intersep = exp(b(1));
epns  = b(2);

% plot the figure
lrDfig = figure;
pos(3)=figWidth; 
pos(4)=figHeight;
set(lrDfig,'color','w','Units','inches','Position',pos)
hold on
errorbar(lrExtract(1:end-1),aveDs(1:end-1),stdDs(1:end-1),'o','MarkerSize',6,'Color',greys(10,:),...
    'MarkerEdgeColor',greys(10,:),'MarkerFaceColor',greys(10,:),'CapSize',0,...
    'LineWidth',2)
plot(lrExtract,intersep*lrExtract.^epns,'Color',greys(6,:),'LineWidth',2)
hold off
box on
% ylim([0.1,5])
xlabel('$\eta$','Interpreter','latex','FontSize',16)
% ylabel('$\langle (\Delta r )^2 \rangle $','Interpreter','latex','FontSize',20)
ylabel('$D$','Interpreter','latex','FontSize',20)
set(gca,'XScale','log','YScale','log','YTick',[1e-4,1e-2,1],'FontSize',16)
% set(gca,'XScale','log','YScale','log','YTick',[1e-1,1],'FontSize',16)


prefix = 'pc_rw_lr_D';
saveas(gcf,[saveFolder,filesep,prefix,'.fig'])
print('-depsc',[saveFolder,filesep,prefix,'.eps'])


%% Diffusion constant vs noise std
dFolder = './data/pc2D_diff_sig';

allFile = dir([dFolder,filesep,'*.mat']);
files = {allFile.name}';

aveDs = nan(length(files),1);  % store the average diffusion constants
stdDs = nan(length(files),1);

rfMass = nan(length(files),2);  % mean and standard deviation of RF mass
nsdExtract = nan(length(files),1);
for i0 = 1:length(files)
    if isempty(files)
        error('folder is empty!')
    else   
%         s1 = '(?<= *sig_)[\d\.]+(?=_)';
        s1 = '(?<= *std)[\d\.]+(?=_)';
        nstd = str2num(char(regexp(files{i0},s1,'match')));
        if ~isempty(nstd)
            nsdExtract(i0) = nstd;
            temp = load(char(fullfile(dFolder,filesep,files{i0})));
            aveDs(i0) = nanmean(temp.Ds);
            stdDs(i0) = nanstd(temp.Ds);
            
            rawMass = temp.pkMas(:);
            rfMass(i0,:) = [nanmean(rawMass(rawMass < 200)),nanstd(rawMass(rawMass < 200))];
        end
    end
end

% log-log fit select partial
sel = 1:5;
logX = [ones(length(nsdExtract(sel)),1),log(nsdExtract(sel))];
logY = log(aveDs(sel));
b = logX\logY;
intersep = exp(b(1));
epns  = b(2);

% plot the figure
sigDfig = figure;
pos(3)=figWidth; 
pos(4)=figHeight;
set(sigDfig,'color','w','Units','inches','Position',pos)

hold on
errorbar(nsdExtract(sel),aveDs(sel),zeros(length(sel),1),stdDs(sel),'o','MarkerSize',6,'Color',greys(10,:),...
    'MarkerEdgeColor',greys(10,:),'MarkerFaceColor',greys(10,:),'CapSize',0,...
    'LineWidth',2)
plot(nsdExtract(sel),intersep*nsdExtract(sel).^epns,'Color',greys(6,:),'LineWidth',2)
hold off
box on
xlim([7e-5,2e-2])
ylim([1e-3,2e-1])
xlabel('$\sigma$','Interpreter','latex','FontSize',20)
% ylabel('$\langle (\Delta r )^2 \rangle $','Interpreter','latex')
ylabel('$D$','Interpreter','latex','FontSize',20)
set(gca,'XScale','log','YScale','log','YTick',[1e-3,1e-2,1e-1],'XTick',[1e-4,1e-3,1e-2],...
    'FontSize',16)

prefix = 'pc2D_grid_std_D';
saveas(gcf,[saveFolder,filesep,prefix,'.fig'])
print('-depsc',[saveFolder,filesep,prefix,'.eps'])


%% Plot the figures
% colors

figure
errorbar(allNps,aveDsAll,stdDsAll,'o-','MarkerSize',10,'Color',greys(10,:),...
    'MarkerEdgeColor',greys(10,:),'MarkerFaceColor',greys(10,:),'CapSize',0,...
    'LineWidth',2)
xlabel('number of place cells')
ylabel('$\langle (\Delta r )^2 \rangle $','Interpreter','latex')
ylabel('$D$','Interpreter','latex')
set(gca,'XScale','log','YScale','linear')


% relative size of place fields, remove the outliners
figure
errorbar(NpExtract,rfMass(:,1),rfMass(:,2),'o','MarkerSize',10,'Color',greys(10,:),...
    'MarkerEdgeColor',greys(10,:),'MarkerFaceColor',greys(10,:),'CapSize',0,...
    'LineWidth',2)
xlabel('number of place cells')
ylabel('Relateive RF Size')
set(gca,'XScale','linear','YScale','linear')

%% Debug
figure
plot(temp.msds(:, randperm(100,5)))

figure
numSel = 3;
hold on
for i = 1:numSel
    pcSel = randperm(size(temp.pkCenterMass,1),1);  % randomly select one place cell
    plot(squeeze(temp.pkCenterMass(pcSel,1,:))/temp.param.ps,...
        squeeze(temp.pkCenterMass(pcSel,2,:))/temp.param.ps,'o','LineWidth',1.5)
end
xlim([0,1])
ylim([0,1])
hold off
box on
legend('neuron 1', 'neuron 2','neuron 3')
title('numer of plac cells : 25')


% ============================================
% Trajectory of example neuron
% ============================================
epInx = randperm(param.Np,1);  % randomly slect
% epInx = 1;

specColors = flip(brewermap(size(xcoord,2), 'Spectral'));

% colors indicate time
figure
hold on
for i=1:size(pks,2)
    plot(xcoord(epInx,i),ycoord(epInx,i),'^','MarkerSize',4,'MarkerEdgeColor',...
        specColors(i,:),'LineWidth',1.5)
end
hold off
box on
xlim([0,32])
ylim([0,32])
xlabel('x position','FontSize',24)
ylabel('y position','FontSize',24)
set(gca,'FontSize',24,'LineWidth',1.5)


% Example msds
selInx = randperm(param.Np,3);
figure
plot(msds(:,selInx))
xlabel('Time','FontSize',24)
ylabel('$\langle (\Delta r)^2\rangle$','Interpreter','latex','FontSize',24)
set(gca,'FontSize',24,'LineWidth',1.5)


% histogram of diffusion constants
figure
histogram(Ds,15)
xlabel('$D$','Interpreter','latex','FontSize',24)
ylabel('Count')


% heatmap of pk amplitude
figure
imagesc(pkAmp,[0,5])
colorbar
xlabel('time')
ylabel('place cell')


%% change of pkMass
figure
plot(temp.pkMas(randperm(temp.Np,3),:)')

numDots = 300;
dotColors = flip(brewermap(numDots,'Spectral'));
figure
hold on
for i = 1:numDots
    plot(posiInfo(i,1)/param.ps,posiInfo(i,2)/param.ps,'+','Color',dotColors(i,:),'MarkerSize',8)
end
hold off
box on
xlim([0,1])
ylim([0,1])
xlabel('x')
ylabel('y')
