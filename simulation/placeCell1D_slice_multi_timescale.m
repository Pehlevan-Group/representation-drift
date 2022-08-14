% using non-negative similarity matching to learn a 1-d place cells based
% on predefined grid cells patterns. Based on the paper lian et al 2020
% 1D grid fields are slices of 2D lattice of grid fields

% Here, each synapse has two timescales, a faster one and a slower one
% we demonstrate that such slow timescale can be used to maintain memories
% even without the constant access of sensory input

%

clear
close all

%% model parameters
param.ps =  200;        % number of positions along each dimension
param.Nlbd = 5;         % number of different scales/spacing
param.Nthe = 6;         % number of rotations
param.Nx =  3;          % offset of x-direction
param.Ny = 3;           % offset of y-direction
param.Ng = param.Nlbd*param.Nthe*param.Nx*param.Ny;   % total number of grid cells
param.Np = 200;         % number of place cells, default 20*20

param.baseLbd = 1/4;    % spacing of smallest grid RF, default 1/4
param.sf =  1.42;       % scaling factor between adjacent module

% parameters for learning 
param.multiTimeScale = true;    % default false, two time scales for the syanpse
noiseStd = 5e-3;        % 0.001
learnRate = 0.05;       % default 0.05

param.W = 0.1*randn(param.Np,param.Ng);   % initialize the forward matrix
param.M = eye(param.Np);        % lateral connection if using simple nsm
param.Wslow = param.W;  % slow forward matrix
param.Mslow = eye(param.Np);    % slow recurrent weight matrix
param.lbd1 = 0.00;              % 0.15 for 400 place cells and 5 modes
param.lbd2 = 0.05;              % 0.05 for 400 place cells and 5 modes


param.alpha = 35;        % the threshold depends on the input dimension, 65
param.beta = 2; 
param.gy = 0.05;         % update step for y
param.gz = 0.1;          % update step for z
param.gv = 0.2;          % update step for V
param.b = zeros(param.Np,1);  % biase
param.learnRate = learnRate;  % learning rate for W and b
param.forgetRate = 1e-6;   % forgetting timescale
param.learnSlow = 1e-3;  % learning rate of the slow timescale
% param.forgetFast = 0;
% param.forgetSlow = 0;
param.noise =  noiseStd; % stanard deivation of noise 
param.rwSpeed = 10;      % steps each update, default 1
param.step = 20;         % store every 20 step
param.ampThd = 0.1;      % amplitude threshold, depending on the parameters

save_date_flag = true;   % store the data or not

param.sigWmax = noiseStd;% the maximum noise std of forward synapses
param.sigMmax = noiseStd;     

param.noiseW =  param.sigWmax*ones(param.Np,param.Ng);    % stanard deivation of noise, same for all
param.noiseM =  param.sigMmax*ones(param.Np,param.Np);   

param.ori = 1/6*pi;     % slicing orientation

param.BatchSize = 1;      % minibatch used to to learn
param.learnType = 'snsm';  % snsm, batch, online, randwalk, direction, inputNoise

gridQuality = 'slice';  % regular, weak or slice

makeAnimation = 0;    % whether make a animation or not

gridFields = slice2Dgrid(param.ps,param.Nlbd,param.Nthe,param.Nx,param.Ny,param.ori);


%% learning and forgetting sessions
% this part mimics the experimental sessions with training and forgetting
param.learnTime = 1e2;
param.forgetTime = 5e2;
param.preTrain = 1e4;   % before doing fortetting and learning
num_sessions = 10;  % we will remove the first 5 session

posiGram = eye(param.ps);
gdInput = gridFields'*posiGram;  % gramin of input matrix

% scenario 1, slow decay term introduced
[output_s, param] = place_cell_learn_forget(gdInput,num_sessions, param);

% scenario 2, slow decay term introduced
param.forgetRate = 1e-4; 
[output_m, param] = place_cell_learn_forget(gdInput,num_sessions, param);

% scenario 3, slow timescale is equal to fast timescale
param.forgetRate = param.learnRate; 
[output_f, param] = place_cell_learn_forget(gdInput,num_sessions, param);


% get rid of the first session
iter_in_session = round((param.learnTime + param.forgetTime)/param.step);  % number of iteration in one session
iter_in_learn = round(param.learnTime/param.step);
iter_in_forget = round(param.forgetTime/param.step);

%% Analysis and plot
sFolder = './figures';

% take the similarity after the first learning session as reference
Ys0_m = output_m(:,:,round(param.learnTime/param.step)+0*iter_in_session);
Ys0_s = output_s(:,:,round(param.learnTime/param.step)+0*iter_in_session);
Ys0_f = output_f(:,:,round(param.learnTime/param.step)+0*iter_in_session);


% SM0 = Ys0_learn'*Ys0_learn;

% D0 = squareform(1 - pdist(Ys0_learn','cosine'));

sm_norm_ratio_s = zeros(size(output_s, 3),1);
sm_norm_ratio_m = zeros(size(output_m, 3),1);
sm_norm_ratio_f = zeros(size(output_f, 3),1);

% for i = 1:size(output,3)
%     Ys = output(:,:,i);
%     Ys2 = output2(:,:,i);
% %     SM = Ys'*Ys;
% %     sm_norm_ratio(i) = norm(SM-SM0)/norm(SM0);
%     D = squareform(1 - pdist(Ys','cosine'));
%     sm_norm_ratio(i) = norm(D-D0)/norm(D0);
%     
%     % cosine similarity for the same time scale
%     D2 = squareform(1 - pdist(Ys2','cosine'));
%     sm_norm_ratio2(i) = norm(D2-D0)/norm(D0);
%     
% end

% another way to show the metric
start_session = 0;
start_point = round(param.learnTime/param.step)+start_session*iter_in_session;  % get rid of the transient
for i = 1:size(output_s,3)
    Ys_s = output_s(:,:,i);
    Ys_m = output_m(:,:,i);
    Ys_f = output_f(:,:,i);
    
    sm_norm_ratio_s(i) = kernelAlignment(Ys0_s,Ys_s);
    sm_norm_ratio_m(i) = kernelAlignment(Ys0_m,Ys_m);
    sm_norm_ratio_f(i) = kernelAlignment(Ys0_f,Ys_f);
end

timepoints= param.step*(1:size(output_s,3));

greys = brewermap(11, 'Greys');
blues = brewermap(11,'Blues');
reds = brewermap(11,'Reds');
plotColors = {blues;reds;greys};

f_sm = figure;
hold on
pos(3)= 4; pos(4)= 3;
set(f_sm,'color','w','Units','inches','Position',pos)
plot(timepoints(start_point:end)' - start_point*param.step, sm_norm_ratio_f(start_point:end),...
    'LineWidth',1.5,'Color', blues(9,:))
plot(timepoints(start_point:end)' - start_point*param.step, sm_norm_ratio_m(start_point:end),...
    'LineWidth',1.5, 'Color', reds(9,:))
plot(timepoints(start_point:end)' - start_point*param.step, sm_norm_ratio_s(start_point:end),...
    'LineWidth',1.5, 'Color', greys(10,:))

for i = 1:num_sessions - start_session-1
    x_shade = [(i*iter_in_session - iter_in_learn)*param.step+1;i*iter_in_session*param.step];
    y_shade = 1.1*ones(size(x_shade));
    ah = area(x_shade(:),y_shade(:),'FaceColor', greys(7,:),'EdgeColor',greys(7,:));
    ah.FaceAlpha = 0.5;
    ah.EdgeAlpha = 0.5;
end

hold off
box on
lg = legend('\eta_{forget} = 0.05','\eta_{forget} = 10^{-3}','\eta_{forget} = 10^{-6}');
set(lg,'FontSize',12)
xlabel('Time','FontSize',16)
% ylabel('Change of cosine similartiy',...
%     'Interpreter', 'latex','FontSize',16)
% ylabel('$|| \mathbf{Y}_0^{\top}\mathbf{Y}_t||_F/(||\mathbf{Y}_0||_F||\mathbf{Y}_t||_F)$',...
%     'Interpreter', 'latex','FontSize',16)
ylabel('Similarity Alignment','FontSize',16)
ylim([0.4, 1.1])
xlim([0,5e3])
set(gca,'YScale','linear','FontSize',14)

prefix = ['changeSM_place1D_slow_timescale_learn_forget_', date];
saveas(gcf,[sFolder,filesep,prefix,'.fig'])
print('-depsc',[sFolder,filesep,prefix,'.eps'])
% save the data

% end of the learning period representations
time_selet = 8*iter_in_session;
ys_all = {output_f(:,:,time_selet);output_m(:,:,time_selet);output_s(:,:,time_selet)};
etas = [0.05;1e-4;1e-6];

figure
for i = 1:3
    subplot(1,3,i)
    imagesc(ys_all{i}'*ys_all{i})
    colormap(viridis)
    colorbar
    set(gca,'Visible','off')
    title(['$\eta = ',num2str(etas(i)),'$'])
end
prefix = 'heatmap_Ys_place1D_slow_timescale_learn';
saveas(gcf,[sFolder,filesep,prefix,'.fig'])
print('-depsc',[sFolder,filesep,prefix,'.eps'])

% ***************************************************************************
% Concantenate all the response in the learning period of every session
% ***************************************************************************
% start from session 10 to remove the transient period
start_session = 1;
learn_time_index = [];
for i = start_session:num_sessions-1
    learn_time_index = [learn_time_index, i*iter_in_session+1:i*iter_in_session+iter_in_learn];
end

Yt_all_learn_s = output_s(:,:,learn_time_index);
Yt_all_learn_m = output_m(:,:,learn_time_index);
Yt_all_learn_f = output_f(:,:,learn_time_index);
Yt_all_learn = {Yt_all_learn_f;Yt_all_learn_m;Yt_all_learn_s};

% measure the drift during training procedure
pvCorr = cell(3,1);
for i = 1:3
    pvCorr{i} = zeros(size(Yt_all_learn_f,3),size(Yt_all_learn_f,2));
end

% pvCorr_f = zeros(size(Yt_all_learn_f,3),size(Yt_all_learn_f,2));
% pvCorr_m = zeros(size(Yt_all_learn_m,3),size(Yt_all_learn_m,2)); 
% pvCorr_s = zeros(size(Yt_all_learn_s,3),size(Yt_all_learn_s,2)); 


for type = 1:3
    for i = 1:size(Yt_all_learn{type},3)
        for j = 1:size(Yt_all_learn{type},2)
            temp = Yt_all_learn{type}(:,j,i);
            C = corrcoef(temp,Yt_all_learn{type}(:,j,1));
            pvCorr{type}(i,j) = C(1,2);
        end
    end
end

f_pvCorr = figure;
set(f_pvCorr,'color','w','Units','inches')
pos(3)=4;  
pos(4)=3;
set(f_pvCorr,'Position',pos)
for i = 1:3
    fh = shadedErrorBar((1:size(pvCorr{i},1))'/iter_in_learn,pvCorr{i}',...
        {@mean,@std});
    hold on
    set(fh.edge,'Visible','off')
    fh.mainLine.LineWidth = 2;
    fh.mainLine.Color = plotColors{i}(10,:);
    fh.patch.FaceColor = plotColors{i}(7,:);
end
lg = legend('\eta_{forget} = 0.05','\eta_{forget} = 10^{-3}','\eta_{forget} = 10^{-6}');
set(lg,'FontSize',12)
box on
xlabel('# Session','FontSize',16)
ylabel('PV correlation','FontSize',16)
% set(gca,'FontSize',16,'LineWidth',1, 'XTick',0:10:20,'XTickLabel',{'0','10^4','2\times 10^4'})
set(gca,'FontSize',16,'LineWidth',1)


prefix = 'PV_corr_place1D_slow_timescale_learn_sessions';
saveas(gcf,[sFolder,filesep,prefix,'.fig'])
print('-depsc',[sFolder,filesep,prefix,'.eps'])
% 

% **********************************************************
% Drift of single RF over time
% **********************************************************
greys = brewermap(11,'Greys');
neuron_example = randperm(param.Np, 3);
xs = 1:param.ps;
sep_plot = 2;  % only take example every 3 steps

f_pv = figure;
pos(3)= 5; pos(4)= 3;
set(f_pv,'color','w','Units','inches','Position',pos)
[ha, ~] = tight_subplot(1,3, [0.02, 0.02], [0.1, 0.10], [0.02,0.01]);

for i = 1:3
    resp_mat = squeeze(Yt_all_learn{2}(neuron_example(i),:,1:sep_plot:end));
    axes(ha(i))
    hold on
    for j = 1:size(resp_mat,2)
        ys = resp_mat(:,j) + (size(resp_mat,2)-j)*0.5;
        plot(xs',ys,'LineWidth',1.5,'Color',greys(8,:))
    end
    hold off
    ha(i).YAxis.Visible = 'off';
    ha(i).YLim = [0, 12];
    xlabel('Position','FontSize',14)
    title(['Neuron ',num2str(i)], 'FontSize', 14)
end


% save the figure
prefix = ['place1D_multitimescale_example_RFs_',date];
saveas(gcf,[sFolder,filesep,prefix,'.fig'])
print('-depsc',[sFolder,filesep,prefix,'.eps'])
%% Debug 
start_session = 0;
% Ys0_s = output_s(:,:,round(param.learnTime/param.step)+0*iter_in_session);

start_point = round(param.learnTime/param.step)+start_session*iter_in_session;  % get rid of the transient
for i = 1:size(output_s,3)
    Ys_s = output_s(:,:,i);
   
    
    sm_norm_ratio_s(i) = kernelAlignment(Ys_s,Ys0_s);
  
end
figure
plot(sm_norm_ratio_s)
ylim([0,1.05])



%% using non-negative similarity matching to learn place fields
% generate input from grid filds

total_iter = 2e4;   % total interation, default 2e3

posiGram = eye(param.ps);
gdInput = gridFields'*posiGram;  % gramin of input matrix

% First the initial stage, only return the updated parameters
[Yt, param] = place_cell_stochastic_update_forget(gdInput,total_iter, param,true, true);
Ys = Yt.Yt(:,:,end);
[pkVal, peakPosi] = sort(Ys,2,'descend');
[~,neuron_order] = sort(peakPosi(:,1),'ascend');

% heatmap without sorting after training
f_pv = figure;
pos(3)= 3.2; pos(4)= 2.5;
set(f_pv,'color','w','Units','inches','Position',pos)
imagesc(Ys,[0,2.5])
colormap(viridis)
colorbar
xlabel('Position','FontSize',16)
ylabel('Neuron sorted','FontSize',16)
set(gca,'XTick',[], 'YTick',[])
title('$t = 0$','Interpreter','latex','FontSize',16)

prefix = [figPre, 'hetmap_place1D_slow_timescale_t0'];
saveas(gcf,[sFolder,filesep,prefix,'.fig'])
print('-depsc',[sFolder,filesep,prefix,'.eps'])

% heatmap of the ordered receptive fields
f_pv = figure;
pos(3)= 3.2; pos(4)= 2.5;
set(f_pv,'color','w','Units','inches','Position',pos)
imagesc(Ys(neuron_order,:), [0,2.5])
colormap(viridis)
colorbar
xlabel('Position')
ylabel('Peuron sorted')
set(gca,'XTick',[], 'YTick',[])
title('$t = 0$','Interpreter','latex')

prefix = [figPre, 'heatmap_sorted_place1D_slow_timescale_t0'];
saveas(gcf,[sFolder,filesep,prefix,'.fig'])
print('-depsc',[sFolder,filesep,prefix,'.eps'])

% ******************************************************************
% representation after 1000 steps on online learning
% ******************************************************************
Ys1st = Yt.Yt(:,:,10);
[pkVal, peakPosi] = sort(Ys1st,2,'descend');
[~,neuron_order] = sort(peakPosi(:,1),'ascend');

f_pv = figure;
pos(3)= 3.2; pos(4)= 2.5;
set(f_pv,'color','w','Units','inches','Position',pos)
imagesc(Ys1st,[0,2.5])
colormap(viridis)
colorbar
xlabel('Position','FontSize',16)
ylabel('Neuron','FontSize',16)
set(gca,'XTick',[], 'YTick',[])
% title('$t = 0$','Interpreter','latex','FontSize',16)

prefix = [figPre, 'hetmap_timescale_learn_t1'];
saveas(gcf,[sFolder,filesep,prefix,'.fig'])
print('-depsc',[sFolder,filesep,prefix,'.eps'])


f_pv = figure;
pos(3)= 3.2; pos(4)= 2.5;
set(f_pv,'color','w','Units','inches','Position',pos)
imagesc(Ys1st(neuron_order,:),[0,2.5])
colormap(viridis)
colorbar
xlabel('Position','FontSize',16)
ylabel('Neuron sorted','FontSize',16)
set(gca,'XTick',[], 'YTick',[])
% title('$t = 0$','Interpreter','latex','FontSize',16)

prefix = [figPre, 'hetmap_sorted_timescale_learn_t1'];
saveas(gcf,[sFolder,filesep,prefix,'.fig'])
print('-depsc',[sFolder,filesep,prefix,'.eps'])


%{
% estimate the place field of place cells
numProb = param.ps;    % number of positions used to estimate the RF

ys_prob = zeros(param.Np,1);
% ys_prob = zeros(param.Np,numProb);
ys = zeros(param.Np, numProb);
for i = 1:numProb
    states= PlaceCellhelper.nsmDynBatch(gdInput(:,i),ys_prob, param);
%     states= PlaceCellhelper.nsmDynBatch(gdInput(:,i),ys_prob, param);
    ys(:,i) = states.Y;
end

% estimate the peak positions of the place field
[~, pkInx] = sort(ys,2,'descend');


pkMat = zeros(1,param.ps);
pkMat(pkInx(:,1)) = 1;

figure
imagesc(pkMat)

[~,nnx] = sort(pkInx(:,1),'ascend');
figure
imagesc(ys(nnx,:))
colormap(viridis)
ylim([150,200])

% amplitude of place fields
z = max(ys,[],2);
figure
histogram(z(z>0))
% xlim([2,6])
xlabel('Amplitude','FontSize',24)
ylabel('count','FontSize',24)
set(gca,'LineWidth',1.5, 'FontSize',24)


% ================= visualize the input ===========================

% input similarity
figure
imagesc(gdInput'*gdInput)
colorbar
xlabel('position index','FontSize',24)
ylabel('position index','FontSize',24)
set(gca,'LineWidth',1.5, 'FontSize',20)
% title('Input similarity matrix','FontSize',24)


% output matrix and similarity matrix
figure
imagesc(ys)
colorbar
xlabel('position index ','FontSize',24)
ylabel('place cell index','FontSize',24)
set(gca,'LineWidth',1, 'FontSize',20)
title('Output','FontSize',24)


figure
imagesc(ys'*ys)
colorbar
xlabel('position index ','FontSize',24)
ylabel('position index ','FontSize',24)
set(gca,'LineWidth',1, 'FontSize',20)
title('Output Similarity','FontSize',24)

% ============== visualize the learned matrix =======================
% feedforward connection
figure
imagesc(param.W)
colorbar
xlabel('grid cell','FontSize',24)
ylabel('place cell','FontSize',24)
set(gca,'LineWidth',1.5,'FontSize',24)

% histogram of W
figure
histogram(param.W(param.W<0.5))
xlabel('$W_{ij}$','Interpreter','latex','FontSize',24)
ylabel('Count','FontSize',24)
set(gca,'LineWidth',1.5,'FontSize',24)

% in term of individual palce cells
figure
plot(param.W(randperm(param.Np,3),:)')


% heatmap of feedforward matrix
figure
imagesc(param.W)
colorbar
xlabel('grid cell index','FontSize',24)
xlabel('place cell index','FontSize',24)
set(gca,'LineWidth',1.5, 'FontSize',20)
title('forward connection matrix','FontSize',24)


% ============== lateral connection matrix ===================
Mhat = param.M - diag(diag(param.M));
figure
imagesc(Mhat)
colorbar
xlabel('place cell index','FontSize',24)
xlabel('place cell index','FontSize',24)
set(gca,'LineWidth',1.5, 'FontSize',20)
title('Lateral connection matrix','FontSize',24)

figure
histogram(Mhat(Mhat>0.005))
xlabel('$M_{ij}$','Interpreter','latex','FontSize',24)
ylabel('Count','FontSize',24)
set(gca,'LineWidth',1.5,'FontSize',24)

% compare the two matrices
% forward matrix
figure
plot(param.W(:), param.Wslow(:),'o')
xlabel('$W_{ij}$','Interpreter','latex','FontSize',24)
ylabel('$W^s_{ij}$','Interpreter','latex','FontSize',24)
set(gca,'LineWidth',1.5,'FontSize',24)

% recurrent matrix

figure
plot(param.M(:), param.Mslow(:),'o')
xlabel('$M_{ij}$','Interpreter','latex','FontSize',24)
ylabel('$M^s_{ij}$','Interpreter','latex','FontSize',24)
set(gca,'LineWidth',1.5,'FontSize',24)
%}
% reference SM
% Ys = Yt.Yt(:,:,end);
SM0 = Ys'*Ys;

% old matrix after learning
Wold = param.W; Mold = param.M; bold = param.b;
%% decay of representation if having the slow synapses
total_iter = 2e4;
time_points = round(total_iter/param.step);

[output, param] = place_cell_stochastic_update_forget(gdInput,total_iter,...
    param,true, false);

%% Analysis and plot
sFolder = ['.',filesep,'figures'];
figPre = ['placeCell_1D_slice_slow_timescale',date];
% showing the tiling of RFs over time
time_sel = [50, 100];
real_time_steps = [10000, 20000];

for i = 1:length(time_sel)
    Ys = output.Yt(:,:,time_sel(i));
    % order neurons based on their tuning position
%     [~,neuron_orde] = sort(Yt.pks(:,time_sel(i)),'ascend');
    [pkVal, peakPosi] = sort(Ys,2,'descend');
    [~,neuron_order] = sort(peakPosi(:,1),'ascend');
    
    % representation unsorted
    f_pv = figure;
    pos(3)= 3.2; pos(4)= 2.5;
    set(f_pv,'color','w','Units','inches','Position',pos)
    imagesc(Ys,[0,2.5])
    colormap(viridis)
    colorbar
    title(['$t =', num2str(real_time_steps(i)),'$'],'Interpreter','latex')
    xlabel(' Position','FontSize',16)
    ylabel('Sorted neuron','FontSize',16)
    set(gca,'XTick',[], 'YTick',[])
    
%     prefix = [figPre, '_heatmap_t', num2str(real_time_steps(i))];
%     saveas(gcf,[sFolder,filesep,prefix,'.fig'])
%     print('-depsc',[sFolder,filesep,prefix,'.eps'])
    
    % representations sorted
    f_pv = figure;
    pos(3)= 3.2; pos(4)= 2.5;
    set(f_pv,'color','w','Units','inches','Position',pos)
    imagesc(Ys(neuron_order,:),[0,2.5])
    colormap(viridis)
    colorbar
    title(['$t =', num2str(real_time_steps(i)),'$'],'Interpreter','latex')
    xlabel(' Position','FontSize',16)
    ylabel('Sorted neuron','FontSize',16)
    set(gca,'XTick',[], 'YTick',[])
    
%     prefix = [figPre, '_heatmap_sorted_t', num2str(real_time_steps(i))];
%     saveas(gcf,[sFolder,filesep,prefix,'.fig'])
%     print('-depsc',[sFolder,filesep,prefix,'.eps'])

    
    % similarity matrix
    figure
    imagesc(Ys'*Ys)
    colormap(viridis)
    colorbar
    title(['$t =', num2str(real_time_steps(i)),'$'],'Interpreter','latex')
    xlabel('Position','FontSize',16)
    ylabel('Position','FontSize',16)
    set(gca,'XTick',[], 'YTick',[])    
    % multidimensional scaling
%     Dist = squareform(pdist(Ys','euclidean'));
%     
%     [Ymds,~] = cmdscale(Dist);
%     figure
%     plot(Ymds(:,1),Ymds(:,2),'.')
%     xlabel('MDS 1')
%     ylabel('MDS 2')
end

% change of the similarity norm
sm_norm_ratio = zeros(size(output.Yt,3),1);
for i = 1:size(output.Yt,3)
    Ys = output.Yt(:,:,i);
    SM = Ys'*Ys;
    sm_norm_ratio(i) = norm(SM-SM0)/norm(SM0);
end

timepoints= param.step*(1:size(output.Yt,3));

f_sm = figure;
pos(3)= 4; pos(4)= 3;
set(f_sm,'color','w','Units','inches','Position',pos)
plot(timepoints', sm_norm_ratio,'LineWidth',1.5)
xlabel('Time','FontSize',16)
ylabel('$|| \mathbf{Y}^{\top}_t\mathbf{Y}_t - \mathbf{Y}^{\top}_0\mathbf{Y}_0||_F^2/||\mathbf{Y}^{\top}_0\mathbf{Y}_0||_F^2$',...
    'Interpreter', 'latex','FontSize',16)
% set(gca,'YScale','linear','FontSize',14,'YLim', [0,1])
set(gca,'YScale','linear','FontSize',14)


prefix = [figPre, 'SM_change_place1D_slow_timescale'];
saveas(gcf,[sFolder,filesep,prefix,'.fig'])
print('-depsc',[sFolder,filesep,prefix,'.eps'])

%% check if the systems have drift
% based on the initial stage
% z1 = Yt.Yt(:,:,10);
% z2 = Yt.Yt(:,:,end);
% figure
% imagesc(z1)
% colormap(viridis)
% colorbar
% xlabel('Position')
% ylabel('neuron')
% title('$t=10^4$', 'Interpreter','latex')
% 
% figure
% imagesc(z2)
% colormap(viridis)
% colorbar
% xlabel('Position')
% ylabel('neuron')
% title('$t=2\times 10^4$', 'Interpreter','latex')

% **********************************************************
% The average correlation coeffient of population vector
% **********************************************************
rawY = Yt.Yt(:,:,101:end);  % the range shoud be case specific
% rawY = output.Yt;
pvCorr = zeros(size(rawY,3),size(rawY,2)); 
% [~,neuroInx] = sort(peakInx(:,inxSel(1)));
blues = brewermap(11,'Blues');


for i = 1:size(rawY,3)
    for j = 1:size(rawY,2)
        temp = rawY(:,j,i);
        C = corrcoef(temp,rawY(:,j,1));
        pvCorr(i,j) = C(1,2);
    end
end

f_pvCorr = figure;
set(f_pvCorr,'color','w','Units','inches')
pos(3)=4;  
pos(4)=3;
set(f_pvCorr,'Position',pos)
fh = shadedErrorBar((1:size(pvCorr,1))'*100,pvCorr',{@mean,@std});
box on
set(fh.edge,'Visible','off')
fh.mainLine.LineWidth = 3;
fh.mainLine.Color = blues(10,:);
fh.patch.FaceColor = blues(7,:);
xlabel('Time','FontSize',16)
ylabel('PV correlation','FontSize',16)
% set(gca,'FontSize',16,'LineWidth',1, 'XTick',0:10:20,'XTickLabel',{'0','10^4','2\times 10^4'})
set(gca,'FontSize',16,'LineWidth',1)

prefix = [figPre, 'PV_corr_place1D_slow_timescale_learn'];
saveas(gcf,[sFolder,filesep,prefix,'.fig'])
print('-depsc',[sFolder,filesep,prefix,'.eps'])


% **********************************************************
% Drift of single RF over time
% **********************************************************
greys = brewermap(11,'Greys');
spectrals = brewermap(20,'Spectral');
neuron_example = randperm(param.Np, 3);
time_sep = 1e3;
xs = 1:param.ps;

f_pv = figure;
pos(3)= 5; pos(4)= 3;
set(f_pv,'color','w','Units','inches','Position',pos)
[ha, ~] = tight_subplot(1,3, [0.02, 0.02], [0.1, 0.10], [0.02,0.01]);

for i = 1:3
%     resp_mat = squeeze(Yt.Yt(neuron_example(i),:,11:end));
    resp_mat = squeeze(Yt_all_learn_s(neuron_example(i),:,:));
    axes(ha(i))
    hold on
    for j = 1:size(resp_mat,2)
        ys = resp_mat(:,j) + (size(resp_mat,2)-j)*0.5;
        plot(xs',ys,'LineWidth',1.5,'Color',greys(8,:))
    end
    hold off
    ha(i).YAxis.Visible = 'off';
    ha(i).YLim = [0, 12];
    xlabel('Position','FontSize',14)
    title(['Neuron ',num2str(i)], 'FontSize', 14)
end

prefix = [figPre, 'RF_drift_slow_timescale'];
saveas(gcf,[sFolder,filesep,prefix,'.fig'])
print('-depsc',[sFolder,filesep,prefix,'.eps'])