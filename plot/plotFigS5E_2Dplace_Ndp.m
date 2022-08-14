% plot the N-dependent effect of 2D place cells

close all
clear

%% setting for the graphics
% plot setting
defaultGraphicsSetttings
blues = brewermap(11,'Blues');
rb = brewermap(11,'RdBu');
set1 = brewermap(9,'Set1');
greys = brewermap(11,'Greys');


sFolder = '../figures';

pos(3)=3.2;  
pos(4)=2.8;
figWidth = 3.2;
figHeight = 2.8;
lineWd = 1.5;
symbSize = 4;
labelSize = 20;
axisSize = 16;
axisWd = 1;
%% Load the data
dFolder = '../data/pc2D_paper_Ndp/';


figPre = 'pc2D';
allFile = dir([dFolder,filesep,'*.mat']);
files = {allFile.name}';
numNs = 10;  % different Ns

s1 = '(?<= *Np_)[\d]+(?=std)';

aveDs = nan(numNs,1);       % store all the mean Ds
stdDs = nan(numNs,1);       % store all the std of Ds
fracActi = nan(numNs,2);
thres  =  0.1;                 % threshold for peak
count = 1;
for i0 = 1:length(files)
    temp = regexp(files{i0},s1,'match');
    if ~isempty(temp)
        Ns(count) = str2num(char(temp));
        raw = load(char(fullfile(dFolder,filesep,files{i0})));

    %     fracActi(i0,:) = [nanmean(raw.pkAmp(:) > thres),nanstd(nanmean(raw.pkAmp > thres,2))];
        aveDs(count,:) = nanmean(raw.Ds);
        stdDs(count,:) = nanstd(raw.Ds);
        count = count + 1;
    end
    
end

[~,Ix] = sort(Ns,'ascend');


%% load the simulation for small N (with repeat)
smallNfile = fullfile(dFolder,'pc2D_smallN_0219.mat');
raw = load(fullfile(smallNfile));
smNs = [1,2,4,8,16];
Ds_sN = nan(length(smNs),2);  % mean and std
for i = 1:length(smNs)
    Ds_sN(i,:) = [nanmean(raw.allDs{i}(:)),nanstd(raw.allDs{i}(:))];
end

% merge small and large N
mergedDs(:,1) = [Ds_sN(:,1);aveDs(Ix(6:end))];
mergedDs(:,2) = [Ds_sN(:,2);stdDs(Ix(6:end))];


% *************************************************
% N-dependent diffusion constants
% ***********************************************
f_N_D= figure;
set(f_N_D,'color','w','Units','inches','Position',pos)

errorbar(Ns(Ix),mergedDs(:,1),mergedDs(:,2),'-o','MarkerSize',symbSize,'MarkerFaceColor',...
    greys(9,:),'MarkerEdgeColor',greys(9,:),'Color',greys(9,:),'LineWidth',...
    lineWd,'CapSize',0)
xlabel('$N$','Interpreter','latex','FontSize',labelSize)
ylabel('$D$','Interpreter','latex','FontSize',labelSize)
set(gca,'FontSize',axisSize,'LineWidth',axisWd,'YScale','log','XScale','log',...
    'YTick',[0.01,0.1,1])

% prefix = [figPre, '_Ns_D_',date];
% saveas(f_N_D,[sFolder,filesep,prefix,'.fig'])
% print('-depsc',[sFolder,filesep,prefix,'.eps'])

