%%Use ONLINE PCA with Hebbian Anti-Hebbian Learning rule (Pehlevan et. al,2015 Neural Computation)
% Test the idea that projected subspace rotate with time

close all
clear

d=10;      % input dimensionality
k=3;      % output dimensionality
n=6001;    % number of samples

% generate the samples
method = 'spiked_covariance_normalized';
% method = 'spiked_covariance';

options_generator=struct;
options_generator.rho=0.01;  % default 0.01
options_generator.lambda_q=1;
% options_generator.gap = 0.2;
% options_generator.slope  = 0.5;
[x,eig_vect,eig_val] = low_rank_rnd_vector(d,k,n,method,options_generator);
[x,eig_vect,eig_val] = standardize_data(x,eig_vect,eig_val);
disp('generated')
%% using FSM algorithm to learn
EPOCHS = 10;  % total epoches to show
sim_metric = 'cosine';
% initialize the algorithm
errors=zeros(n,1)*NaN;
Uhat0 = bsxfun(@times,x( :,1:k), 1./sqrt(sum(x(:,1:k).^2,1)))';
scal = 100;
Minv0 = eye(k) * scal;
Uhat0 = Uhat0 / scal;

tau = 0.1;
learning_rate = 5e-4;
fsm = SSM(k, d, tau, Minv0, Uhat0, learning_rate);

% set the noisy update
noise_level = 0;

% selective store some of the elements
num_sel = 100;
inx_sel = randperm(n,num_sel);      % randomly select 100 samples
all_psp = cell(EPOCHS,1);       % store the projection of different epochs
all_indx = cell(EPOCHS,1);      % store all the index after permutation
feature_mat = cell(EPOCHS,1);   % feature map matrix
psp_comp = nan(k,n);
psp_last_epoch = cell(n,1);   % storet the projected representaiton of last epoch
psp_error = [];

% outer loop, permute the order streaming samples
for j = 1:EPOCHS
    perm_inx = randperm(n);
    all_indx{j} = perm_inx;
    for i = 1:n
        if mod(i,50) == 1
            errors(i,:) = compute_projection_error(eig_vect, fsm.get_components([]));
            psp_error = [psp_error,errors(i,:)];
        end
        
        
    %fsm.fit_next(x(:,i)');
    
%     if j <= 10
%         fsm.fit_next(x(:,perm_inx(i))');
%     else
%         fsm.fit_next_noisy(x(:,perm_inx(i))',noise_level);
%     end
    fsm.fit_next_noisy(x(:,perm_inx(i))',noise_level);
    if j==EPOCHS
        psp_last_epoch{i} = fsm.Minv\(fsm.W*x(:,inx_sel));
    end
%     psp_comp(:,i)= fsm.fit_next(x(:,i)');
    %all_psp{j}(:,i) = fsm.fit_next(x(:,i)');
%     all_psp{j}(:,perm_inx(i)) = fsm.fit_next(x(:,perm_inx(i))');
    end
    
    
%     all_psp{j} = fsm.Minv*fsm.W*x;
    all_psp{j} = fsm.Minv\(fsm.W*x(:,inx_sel));
    feature_mat{j} = fsm.Minv\fsm.W;
end
loglog(1:n,errors,'.')
psp_comp = fsm.get_components([]);

% only interested in selected points, for example, first 9 points
psp_select = cell(n,1);
perm_inx = randperm(n);
allrepres = cell(n,1);
all_sim = cell(n,1);
for i = 1:n
    psp_select{i} = psp_last_epoch{i}(:,1:9);
    temp = psp_last_epoch{i}(:,1:9);
    if strcmp(sim_metric, 'cosine')
        %D = pdist(temp','cosine');
        %all_sim{i} = 1 - squareform(D);
        all_sim{i} = pdist(temp','cosine');
    elseif strcmp(sim_metric, 'sm')
        all_sim{i} = temp' * temp;
    end
%all_psp{j}(:,perm_inx(i)) = fsm.fit_next(x(:,perm_inx(i))');
end

figure
plot(1:n,errors,'o','MarkerSize',10,'MarkerFaceColor','k','MarkerEdgeColor','k')
xlabel('T', 'FontSize',28)
ylabel('subspace error', 'FontSize',28)
set(gca,'YScale', 'log','XScale', 'log','LineWidth',1.5,'FontSize',24)

%% Show how the representation change with time
% randomly select 500 point to view
% define the color
mycolors = brewermap(9,'Set1');
blues = brewermap(11,'Blues');
RedBlues = brewermap(15,'RdBu');
num_show = 100;
indx = randperm(n,num_show);

% plot every 10 epoches
% h1 = figure;
% for inx = 1:10
% %     inx = 20*i;
%     subplot(3,4,inx)
%     hold on
%     scatter3(all_psp{inx}(1,indx(5:end)),all_psp{inx}(2,indx(5:end)),all_psp{inx}(3,indx(5:end)),...
%         'MarkerEdgeColor','k')
%     for j0 = 1:5
%         scatter3(all_psp{inx}(1,indx(j0)),all_psp{inx}(2,indx(j0)),all_psp{inx}(3,indx(j0)),...
%             'MarkerEdgeColor',mycolors(j0,:),'MarkerFaceColor', mycolors(j0,:))
%     end
% end


% plot the projection of first 9 points
% h2 = figure;
% inx = 1 + (0:500:5000);
% for i = 1:9
%     subplot(3,3,i)
%     for j0 = 1:9
%         scatter3(psp_select{inx(i)}(1,j0),psp_select{inx(i)}(2,j0),psp_select{inx(i)}(3,j0),...
%             'MarkerEdgeColor',mycolors(j0,:),'MarkerFaceColor', mycolors(j0,:))
%         hold on
%     end
%     title(['iter ', num2str(inx(i))])
%     set(gca,'FontSize',12)
% end
% hold off

% ******************************************************************
% plot the trajectory of first three points
% ******************************************************************
trajects = cell(3,1);
for i = 1:3
    trajects{i} = nan(n,k);
end
for j = 1:3
    for i = 1:n
        trajects{j}(i,:) = psp_last_epoch{i}(:,j)';
    end
end

figure
for i = 1:3
    plot3(trajects{i}(:,1),trajects{i}(:,2),trajects{i}(:,3),'LineWidth',1,'Color',mycolors(i,:))
    hold on
end
title(['psp, leaning rate:',num2str(learning_rate)])
xlabel('pc1','FontSize',28)
ylabel('pc2','FontSize',28)
zlabel('pc3','FontSize',28)
set(gca,'FontSize',24,'LineWidth',1.5)

% time dependent change of representation
figure
tj_sel = randperm(10,1);
for i = 1:3
    plot((1:n)',trajects{i}(:,tj_sel),'LineWidth',1,'Color',mycolors(i,:))
    hold on
end
hold off
title(['psp, leaning rate:',num2str(learning_rate)])
xlabel('time','FontSize',28)
ylabel('value','FontSize',28)
set(gca,'FontSize',24,'LineWidth',1.5)


% ******************************************************************
% heatmap of selected patterns
% ******************************************************************
% heat map
figure
subplot(3,1,1)
imagesc(psp_select{3000})
title('iteration 3000')
colorbar
% figure
subplot(3,1,2)
imagesc(psp_select{4000})
title('iteration 4000')
colorbar

% figure
subplot(3,1,3)
imagesc(psp_select{5000})
colorbar
title('iteration 5000')

% ******************************************************************
% pair-wise similarity matrices, random slect
% ******************************************************************
% 
figure
inx = (2:5)*1000;
for i = 1:length(inx)
    subplot(2,2,i)
    sm = 1- squareform(all_sim{inx(i)});
    imagesc(sm)
    colorbar
    title(['cosine, iteration ',num2str(inx(i))], 'FontSize',16)
end

% trajectories of pair-wise similarity
pair_sm = nan(4000,length(all_sim{1}));
for i = 1001:5000
    pair_sm(i-1000,:) = 1 - all_sim{i};
end
figure
plot((1:4000)'+1000,pair_sm(:,randperm(36,5)),'LineWidth',1.5)
xlabel('T', 'FontSize',28)
ylabel('cosine similarity', 'FontSize',28)
set(gca,'LineWidth',1.5,'FontSize',24)



% How does the correlation change with time
% figure
% plot3(pair_sm(:,1),pair_sm(:,2),pair_sm(:,3),'LineWidth',1,'Color',mycolors(1,:))

% figure
% histogram(pair_sm(:,4),40)


% autocorr(pair_sm(:,9),'NumLags',400)

% ******************************************************************
% how representation change cross epochs and within 
% ******************************************************************
corr_type = 'cosine';  % or cosine
resp_corr = nan(num_sel,EPOCHS);
for i = 1:EPOCHS
    for j = 1:num_sel
        resp_corr(j,i) = 1-pdist([all_psp{1}(:,j),all_psp{i}(:,j)]',corr_type);
    end
end
mean_corr = mean(resp_corr,1);
std_corr = std(resp_corr,0,1);

% plot individual and average
figure
hold on
plot((1:EPOCHS)',resp_corr(1:10,:)','-','Color',blues(4,:),'LineWidth',1.5)
errorbar((1:EPOCHS)',mean_corr',std_corr','o-','MarkerSize',12,'MarkerFaceColor',...
    blues(9,:),'MarkerEdgeColor',blues(9,:),'Color',blues(9,:),'LineWidth',3)
hold off
title(['psp, leaning rate:',num2str(learning_rate)])
% ylim([0.5,1])
xlabel('epochs', 'FontSize',28)
ylabel('similarity', 'FontSize',28)
set(gca,'LineWidth',1.5,'FontSize',24)

% representational similarity cross different poch
sel_inx = randperm(num_sel,10);  % random select 10
sm_epoch = nan(10*(10-1)/2,EPOCHS);
for i = 1:EPOCHS
    sm_epoch(:,i) = 1-pdist(all_psp{i}(:,sel_inx)',corr_type);
end

% change of pair-wise similarity across epoch
figure
plt_sel = randperm(45,7);
set1 = brewermap(9,'Set1');
for i = 1:length(plt_sel)
    plot(1:EPOCHS,sm_epoch(plt_sel(i),:),'o-','LineWidth',2,'MarkerSize',8,...
        'MarkerEdgeColor',set1(i,:),'MarkerFaceColor',set1(i,:),'Color',set1(i,:))
    hold on
end
ylim([-1,1])
xlabel('epoch', 'FontSize',28)
ylabel('pairwise similarity', 'FontSize',28)
set(gca,'LineWidth',1.5,'FontSize',24)



% representational similarity within last epoch
resp_sm_one_epoch = nan(n,num_sel);
for i = 1:num_sel
    for j = 1:n
        resp_sm_one_epoch(j,i) = 1-pdist([psp_last_epoch{1}(:,i),psp_last_epoch{j}(:,i)]',...
            corr_type);
    end
end

% auto_corr = nan(n,10);
repr_matr_last = cell(10,1);
for j = 1:10
    for i = 1:n
        repr_matr_last{j}(:,i) = psp_last_epoch{i}(:,j);
    end
end
num_steps = 500;
auto_corr = nan(k,num_steps+1);  % store the auto-correlation coefficient

for i = 1:k
    [auto_corr(i,:),auto_lags,~] = autocorr(repr_matr_last{1}(i,:),'NumLags',num_steps);
end



% randomply slect 10 samples
sel_plot = randperm(num_sel,20);
mean_last_epoch = mean(resp_sm_one_epoch,2);
std_last_epoch = std(resp_sm_one_epoch,0,2);

figure
hold on
plot((1:n)',resp_sm_one_epoch(:,sel_plot),'-','Color',blues(4,:),'LineWidth',1.5)
plot((1:n)',mean_last_epoch,'Color',blues(9,:),'LineWidth',3)
% errorbar((1:n)',mean_last_epoch,std_last_epoch','o-','MarkerSize',12,'MarkerFaceColor',...
%     blues(9,:),'MarkerEdgeColor',blues(9,:),'Color',blues(9,:),'LineWidth',3)
hold off
title(['psp, leaning rate:',num2str(learning_rate)])
% ylim([0.5,1.1])
xlabel('T', 'FontSize',28)
ylabel('similarity', 'FontSize',28)
set(gca,'LineWidth',1.5,'FontSize',24)


% plot the autocorrelation of one representation
figure
plot(auto_lags(1:100)',auto_corr(:,1:100)','Color',blues(6,:))
title(['psp, leaning rate:',num2str(learning_rate)])
xlabel('$\tau$','Interpreter','latex', 'FontSize',28)
ylabel('auto corr. effi.', 'FontSize',28)
set(gca,'LineWidth',1.5,'FontSize',24)



%****************************************************
% Change of the feature map matrix
%****************************************************
num_plot = 5;
sel_inx = randperm(d*k,num_plot);
feature_corr = nan(k,EPOCHS);

for i = 1:k
    for j = 1:EPOCHS
        feature_corr(i,j) = 1- pdist([feature_mat{1}(i,:);feature_mat{j}(i,:)],corr_type);
    end
end
mean_feature_corr = mean(feature_corr,1);
std_feature_corr = std(feature_corr,0,1);

figure
hold on
plot((1:EPOCHS)',feature_corr','-','Color',blues(4,:),'LineWidth',1.5)
errorbar((1:EPOCHS)',mean_feature_corr',std_feature_corr','o-','MarkerSize',12,'MarkerFaceColor',...
    blues(9,:),'MarkerEdgeColor',blues(9,:),'Color',blues(9,:),'LineWidth',3)
hold off
title(['feature map, leaning rate:',num2str(learning_rate)])
% ylim([0.5,1])
xlabel('epochs', 'FontSize',28)
ylabel('feature similarity', 'FontSize',28)
set(gca,'LineWidth',1.5,'FontSize',24)

% heat map of the feature map
F1 = feature_mat{1};
F2 = feature_mat{end};
figure
subplot(1,2,1)
imagesc(F1 * F1')
title('$FF^{\top}$','Interpreter','latex')
set(gca,'FontSize',16)
colorbar
subplot(1,2,2)
imagesc(F2 * F2')
title('$FF^{\top}$','Interpreter','latex')
set(gca,'FontSize',16)
colorbar

% change of feature maps across different epoch
figure
plt_sel = randperm(50,5);
set1 = brewermap(7,'Set1');
for i = 1:length(plt_sel)
    plot(1:EPOCHS,feature_mat{i}(:,plt_sel(i)),'o-','LineWidth',2,'MarkerSize',8,...
        'MarkerEdgeColor',set1(i,:),'MarkerFaceColor',set1(i,:),'Color',set1(i,:))
    hold on
end
hold off
% ylim([-1,1])
title('feature map')
xlabel('epoch', 'FontSize',28)
ylabel('$F_{ij}$','Interpreter','latex','FontSize',28)
set(gca,'LineWidth',1.5,'FontSize',24)

% psp error
figure
plot((1:length(psp_error))/120,psp_error,'LineWidth',2)
xlabel('epoch','FontSize',28)
ylabel('PSP eorr','FontSize',28)
set(gca,'YScale','log','FontSize',24,'LineWidth',1.5)























