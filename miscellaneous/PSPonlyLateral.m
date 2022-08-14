% This program explores the idea that, if only lateral connection matrix M
% is noisy, the learned representation drifts differently
% this is script is modified from "driftAnalysis.m"
% First, we fixed the value of W as the desired offline solution
% Second, we set W to be random bust sparse matrix
% last revised on 9/27/2021

clear
close all

%%
saveFolder = './figures';
type = 'psp';           % 'psp' or 'expansion
tau = 0.5;              %  scaling of the learning rate for M
learnType = 'online';  %  online, offline, batch   

% plot setting
%defaultGraphicsSetttings
blues = brewermap(11,'Blues');
rb = brewermap(11,'RdBu');
set1 = brewermap(9,'Set1');

%% learning rate dependent, keep eigen spectrum and noise level, tau the same
n = 10;
k = 3;
eigens = [4.5,3.5,1,ones(1,n-3)*0.01];
% eigens = [5,4,0.4,ones(1,7)*0.05];
% eigens = [1.2 1.1 0.8];
noiseStd = 0.005; % 0.005 for expansion, 0.1 for psp
learnRate = 0.05;
sp = 1;   % the sparsty of input matrix


t = 2e3;   % total number of iterations
% step = 20;  %storing every 20 step
% num_store = ceil(t/step);

% generate input data, first covariance matrix
V = orth(randn(n,n));
U = orth(randn(k,k));       % arbitrary orthogonal matrix to determine the projection
C = V*diag(eigens)*V'/sqrt(n);    % with normalized length, Aug 6, 2020
Vnorm = norm(V(:,1:k));

% generate multivariate Gaussian distribution with specified correlation
% matrix
X = mvnrnd(zeros(1,n),C,t)';
Cx = X*X'/t;      % input sample covariance matrix

% generate some test stimuli that are not used during learning
num_test = 1e2;
Xtest = mvnrnd(zeros(1,n),C,num_test)';
% Xtest = randn(n,num_test) +  2;  % assume to be different distribution

% ========================================================
% Use offline learning to find the inital solution
% ========================================================
dt = learnRate;
W = randn(k,n);
M = eye(k,k);

% define a random mask matrix
mask = rand(k,n) < sp;
% maskM = rand(k,k) < sp;
% F = pinv(M)*W; % inital feature map
noise1 = noiseStd;
noise2 = noiseStd;
dt0 = 0.1;          % learning rate for the initial phase, can be larger for faster convergence
tau0 = 0.5;
pspErrPre = nan(100,1); % store the psp error during the pre-noise stage
% for i = 1:100
%     Y = pinv(M)*W*X; 
%     W = (1-2*dt0)*W + 2*dt0*Y*X'/t;
% %     W = (1-2*dt0)*W + 2*dt0*Y*X'/t + sqrt(2*dt)*noise1*randn(k,n);
%     M = (1-dt0/tau0)*M + dt0/tau0*(Y*Y')/t;
%     F = pinv(M)*W;
%     disp(norm(F*F'-eye(k),'fro'))
% %     pspErrPre(i) = norm(F'*F - V(:,1:k)*V(:,1:k)');
% end

% display psp error

% define the read out
% artificial desired output
labels = randn(1,t);
actiFun = 'linear';
% Wout = labels*Y'/t;   % hebbian readout

%% now add noise to see how F and output drift

tot_iter = 1e5;     % total number of updates
num_sel = 200;      % randomly selected samples used to calculate the drift and diffusion constants
step = 10;          % store every 10 updates
time_points = round(tot_iter/step);
Yt = zeros(k,time_points,num_sel);
Ytest = zeros(k,time_points,num_test);  % store the testing stimulus

Ft = zeros(k,n,time_points);        % store the feature maps
inx = randperm(t);
sel_inx = randperm(t,num_sel);    % samples indices
% Wout = zeros(time_points+1,k);        % store the dynamic read out
Wout = randn(1,k);   % readout vector
readError = nan(tot_iter,1);      % store the average instantaneous readout error

Wout_store = nan(time_points,k);

eigVecs = nan(k,k,time_points);   % eigenvectors of output

% store all the representations at three time points
subPopu = 1000;             % number of subpopulation chosen
ensembStep = 1e4;           % store all the representation every ensembStep
partial_index = randperm(t,subPopu);   
Y_ensemble = zeros(k,subPopu,round(tot_iter/ensembStep) + 1);    %store a subpopulation
pspErr = nan(time_points,1);
featurMapErr = nan(time_points,1);
SMerror = nan(time_points,1);
% use online or offline learning
% evsOld = eye(k);

% X0 = F*X(:,sel_inx);
% SM0 = X0'*X0;   % similarity matrix at time 0
% W = mask.*W;     % use the mask matrix to make the connection random
W = randn(k,n)/sqrt(n);
W = mask.*W;
% M = maskM.*M;   % add mask to M
if strcmp(learnType,'offline')
    for i = 1:tot_iter
        % sample noise matrices
        xis = randn(k,n);
        zetas = randn(k,k);
        Y = pinv(M)*W*X; 
    %     W = (1-2*dt)*W + 2*dt*(Y*X'/t + noise*randn(k,n));
    %     M = (1-dt/tau)*M + dt/tau*(Y*Y'/t +  noise*randn(k,k));
%         W = (1-2*dt)*W + 2*dt*Y*X'/t + sqrt(2*dt)*noise1*xis;
        M = (1-dt/tau)*M + dt/tau*Y*Y'/t +  sqrt(dt/tau)*noise2*zetas;
        
        % readout matrix and error
        Wout = Wout + dt*(labels - Wout*Y)*Y'/t;
        readError(i) = mean((labels - Wout*Y).^2);      
        
        if mod(i,step)==0
            temp = pinv(M)*W;       % current feature map
            Yt(:,round(i/step),:) = temp*X(:,sel_inx);
            Ft(:,:,round(i/step)) = temp; % store
            Wout_store(round(i/step),:) = Wout;
           
            % PSP error
            if strcmp(type,'psp')
                pspErr(round(i/step)) = norm(temp'*temp - V(:,1:k)*V(:,1:k)');
            end
            featurMapErr(round(i/step)) = norm(temp*temp' - eye(k));
            Ysel = temp*X(:,sel_inx);
            SMerror(round(i/step)) = norm(Ysel'*Ysel - SM0,'fro')/(norm(SM0,'fro') + 1e-10);
            % representation of test stimuli
            Ytest(:,round(i/step),:) = temp*Xtest;
         
        end

        % store population representations every 10000 steps
        if mod(i, ensembStep)==0 || i == 1
           Y_ensemble(:,:,round(i/1e4)+1) = pinv(M)*W*X(:,partial_index);
        end
    
    end
elseif strcmp(learnType, 'SDE')
    % update the similified SDE of M and F
    % define the noise matrix
    for i = 1:tot_iter
        xis = randn(k,n);
        zetas = randn(k,k);
        M = M + 2*dt*(F*Cx*F' - M) + noise1*sqrt(dt)*zetas;
        F = F + 2*dt*pinv(M)*F*Cx*(eye(n) - F'*F)+ sqrt(dt)*pinv(M)*(noise1*xis - noise2*zetas*F);
        if mod(i,step)==0
            Yt(:,round(i/step),:) = F*X(:,sel_inx);
            Ft(:,:,round(i/step)) = F; % store
            
            % representation of test stimuli
            Ytest(:,round(i/step),:) = F*Xtest;
            
            % eigvectors
            Y = F*X;
%             evs = eigens(Y*Y');
            evs = pca(Y');
            eigVecs(:,:,round(i/step)) = evs;   
        end
    end
elseif strcmp(learnType,'online')
    for i = 1:tot_iter
        curr_inx = randperm(t,1);  % randm generate one sample
        % generate noise matrices
        xis = randn(k,n);
        zetas = randn(k,k);
        Y = pinv(M)*W*X(:,curr_inx); 
%         W = (1-2*dt)*W + 2*dt*Y*X(:,curr_inx)' + sqrt(dt)*noise1*xis;
%         W = mask.*W;  % add mask
        M = (1-dt/tau)*M + dt/tau*Y*Y' +  sqrt(dt)*noise2*zetas;
%         M = maskM.*M;  % mask
        
        % readout error 
%         Wout = Wout + 0.2*dt*(labels(curr_inx) - Wout*Y)*Y';
        z = Wout*Y;
        Wout = Wout + 0.2*dt*(Y' - z*Wout)*z;
%         readError(i) = mean((labels - Wout*pinv(M)*W*X).^2);
        
        if mod(i,step)==0
            temp = pinv(M)*W;  % current feature map
            Yt(:,round(i/step),:) = temp*X(:,sel_inx);
            Ft(:,:,round(i/step)) = temp; % store
            Wout_store(round(i/step),:) = Wout;
                
                % PSP error
            if strcmp(type,'psp')
                pspErr(round(i/step)) = norm(temp'*temp - V(:,1:k)*V(:,1:k)');
            end
            featurMapErr(round(i/step)) = norm(temp*temp' - eye(k));
            Ysel = temp*X(:,sel_inx);
%             SMerror(round(i/step)) = norm(Ysel'*Ysel - SM0,'fro');
            % representation of test stimuli
            Ytest(:,round(i/step),:) = temp*Xtest;
                    
        end
        
        % store population representations every 10000 steps
        if mod(i, ensembStep)==0 || i == 1
           Y_ensemble(:,:,round(i/1e4)+1) = pinv(M)*W*X(:,partial_index);
        end     
    end
end


%% Analysis

% =============== Property of Feature map =================
Fvec = reshape(Ft,k*n,time_points);

% correlation matrix of the feature map
% a = reshape(Ft,k*n,timep);
figure
imagesc(cov(Fvec)); colorbar;
title('Cov(Ft)')
xlabel('time','FontSize',24)
ylabel('time','FontSize',24)
set(gca,'FontSize',20,'LineWidth',1.5)


% FF^t deviation from identity matrix
normFFtErr = nan(time_points,1);
for i = 1:time_points
    normFFtErr(i) = norm(Ft(:,:,i)*Ft(:,:,i)'- eye(k),'fro');
end
figure
plot((1:time_points)'*step, normFFtErr)
xlabel('iteration')
ylabel('$||FF^{\top} - I_{k}||_F$','Interpreter','latex')

% plot the auto correlation function of the vectorized feature map
CM = corrcoef(Fvec);
imagesc(CM); colorbar;
figure
plot(CM(1000,:))

%%

% plot the ensemble representations
figYens = figure;
set(figYens,'Units','inches','Position',[0,0,4,3])
% [k, N, S] = size(Y_ensemble);
ensem_sel = [10, 11];   % this should depdend on the learning type
for i = 1:length(ensem_sel)
    scatterHd = plot3(Y_ensemble(1,:,ensem_sel(i)),Y_ensemble(2,:,ensem_sel(i)),...
        Y_ensemble(3,:,ensem_sel(i)),'.','MarkerSize',6);
    hold on
    scatterHd.MarkerFaceColor(4) = 0.2;
    scatterHd.MarkerEdgeColor(4) = 0.2;
    grid on
end
xlabel('$y_1$','Interpreter','latex','FontSize',20)
ylabel('$y_2$','Interpreter','latex','FontSize',20)
zlabel('$y_3$','Interpreter','latex','FontSize',20)
set(gca,'FontSize',16)

% based on testing data
figYens = figure;
set(figYens,'Units','inches','Position',[0,0,4,3])
% [k, N, S] = size(Y_ensemble);
ensem_sel = [5e3, 1e4];   % this should depdend on the learning type
for i = 1:length(ensem_sel)
    scatterHd = plot3(squeeze(Ytest(1,ensem_sel(i),:)),squeeze(Ytest(2,ensem_sel(i),:)),...
        squeeze(Ytest(3,ensem_sel(i),:)),'.','MarkerSize',6);
    hold on
    scatterHd.MarkerFaceColor(4) = 0.2;
    scatterHd.MarkerEdgeColor(4) = 0.2;
    grid on
end
xlabel('$y_1$','Interpreter','latex','FontSize',20)
ylabel('$y_2$','Interpreter','latex','FontSize',20)
zlabel('$y_3$','Interpreter','latex','FontSize',20)
set(gca,'FontSize',16)


% save the figure
% prefix = [type,'_ensemble_orientation'];
% saveas(figYens,[sFolder,filesep,prefix,'.fig'])
% print('-depsc',[sFolder,filesep,prefix,'.eps'])


% plot the evolution of Euler angles
% angles = SMhelper.eluerAngles(eigVecs);
% [diffConst, alphas] = SMhelper.diffConstAngles(angles,step,1000);


% show the rotational meansqure displacement and fit a linear
% if expansion, take a look at the projcted dynamics

% Yproj = nan(n,time_points,size(Ytest,3));
% for i = 1:size(Ytest,3)
% %     [evs, D] = eig(Ytest(:,:,i)*Ytest(:,:,i)'/time_points);
%     evs = pca(Ytest(:,:,i)');
%     Yproj(:,:,i) = evs(:,1:n)'*Ytest(:,:,i);
% end
%             evs = pca(Y');
%             evs = SMhelper.newEigVec(evs,evsOld);  % align the eigenvectors

% ======================Rotational diffusion constant ==============
% if strcmp(type,'expansion')
%     rmsd = SMhelper.rotationalMSD(Yproj);
% else
%     rmsd = SMhelper.rotationalMSD(Yt);
% end
% rmsd = SMhelper.rotationalMSD(Yt);
% [dph,ephi] = SMhelper.fitRotationDiff(rmsd,step,2000);

% [msd_tot,msd_comp] = SMhelper.rotationalMSD(Yt,eigVecs);
% dphs = nan(k,2);
% for i= 1:k
%     rmsd = squeeze(msd_comp(i,:,:));
%     [dph0,ephi0] = SMhelper.fitRotationDiff(rmsd,step, 500);
%     dphs(i,:) = [dph0,ephi0];
% end

% eigenvalues and eigen vectors
[V1,D1] = eig(Y_ensemble(:,:,2)*Y_ensemble(:,:,2)'/2000);
[V2,D2] = eig(Y_ensemble(:,:,3)*Y_ensemble(:,:,3)'/2000);

% Hebbian read out
Z1 = Wout_store(1000,:)*Y_ensemble(:,:,1);
Z2 = Wout_store(5000,:)*Y_ensemble(:,:,5);

Z_true = labels(partial_index);

% Hebbian readout with updated weight
hbh = figure;
set(gcf,'color','w','Units','inches')
pos(3)=4.6;  %17.4; %8.5;%pos(3)*1.5;
pos(4)=3.8;%pos(4)*1.5;
set(gcf,'Position',pos)


plot(Z1,Z2,'.')
xlabel('$t = 1$','Interpreter','latex','FontSize',24)
ylabel('$t = 5\times 10^4$','Interpreter','latex','FontSize',24)
set(gca,'FontSize',20,'LineWidth',1.5)
xlim([-4.5,4.5])
ylim([-4.5,4.5])

prefix = 'oja_readout_fig2';
saveas(hbh,[saveFolder,filesep,prefix,'.fig'])
print('-depsc',[saveFolder,filesep,prefix,'.eps'])

%% pair-wise similarity
ix = 1:10:ceil(tot_iter/step);
sm = nan(10,10,length(ix));
for i = 1:10
    for j = 1:10
        for k0 = 1:length(ix)
            sm(i,j,k0) = Yt(:,ix(k0),i)'*Yt(:,ix(k0),j)./norm(Yt(:,ix(k0),i)','fro')/norm(Yt(:,ix(k0),j),'fro');
        end
    end
end

S1 = sm(:,:,100);S2 = sm(:,:,200);
figure
subplot(1,2,1)
imagesc(S1);colorbar
title('iteration:1e4','FontSize',24)
xlabel('sample index','FontSize',20)
ylabel('sample index','FontSize',20)
set(gca,'FontSize',20,'LineWidth',1.5)
subplot(1,2,2)
imagesc(S2);colorbar
title('iteration:2e4','FontSize',24)
xlabel('sample index','FontSize',20)
ylabel('sample index','FontSize',20)

%% pair-wise similarity for the testing stimulus
ix = 1:10:ceil(tot_iter/step);
sm = nan(20,20,length(ix));
for i = 1:20
    for j = 1:20
        for l0 = 1:length(ix)
            sm(i,j,l0) = Ytest(:,ix(l0),i)'*Ytest(:,ix(l0),j)./norm(Ytest(:,ix(l0),i)','fro')/norm(Ytest(:,ix(l0),j),'fro');
        end
    end
end

S1 = sm(:,:,100);S2 = sm(:,:,200);
figure
subplot(1,2,1)
imagesc(S1);colorbar
title('iteration:1e4','FontSize',24)
xlabel('sample index','FontSize',20)
ylabel('sample index','FontSize',20)
set(gca,'FontSize',20,'LineWidth',1.5)
subplot(1,2,2)
imagesc(S2);colorbar
title('iteration:2e4','FontSize',24)
xlabel('sample index','FontSize',20)
ylabel('sample index','FontSize',20)
set(gca,'FontSize',20,'LineWidth',1.5)

%% 
figure
plot((8e3:size(Ytest,2))'*step,Ytest(1:3,8e3:end,23)')
xlabel('$ t$','Interpreter','latex','FontSize',28)
ylabel('$ y_i$','Interpreter','latex','FontSize',28)
set(gca,'FontSize',24,'LineWidth',1.5)

%% Plot and save the figures
% first define all the colors might be used 

% divergent colors, used in heatmap
nc = 256;   % number of colors
spectralMap = brewermap(nc,'Spectral');
PRGnlMap = brewermap(nc,'PRGn');
RdBuMap = flip(brewermap(nc,'RdBu'),1);

blues = brewermap(11,'Blues');
BuGns = brewermap(11,'BuGn');
greens = brewermap(11,'Greens');
greys = brewermap(11,'Greys');
PuRd = brewermap(11,'PuRd');
YlOrRd = brewermap(11,'YlOrRd');
oranges = brewermap(11,'Oranges');

set1 = brewermap(11,'Set1');


%% ================= Figure 2 =========================
% layout of the figure
% 2 x 3 pannels
sFolder = './figures';
paperFig2 = figure; 
set(gcf,'color','w','Units','inches')
pos(3)=14;  %17.4; %8.5;%pos(3)*1.5;
pos(4)=8;%pos(4)*1.5;
set(gcf,'Position',pos)
labelFontSize = 24;
gcaFontSize = 20;


% PSP error
aAxes = axes('position',[.08  .63  0.25  0.35]); hold on
annotation('textbox', [.005 .98 .03 .03],...
    'String', 'A','BackgroundColor','none','Color','k',...
    'LineStyle','none','fontsize',labelFontSize,'HorizontalAlignment','Center');

% example trajectories
bAxes = axes('position',[.4  .63  .25  .35]);
annotation('textbox', [.33 .98 .03 .03],...
    'String', 'B','BackgroundColor','none','Color','k',...
    'LineStyle','none','fontsize',labelFontSize,'HorizontalAlignment','Center');


% similarity matrix
cAxesii = axes('position',[.8  .63  .15  .15]); hold on
cAxesi = axes('position',[.8  .8  .15  .15]); hold on

annotation('textbox', [.68 .98 .03 .03],...
    'String', 'C','BackgroundColor','none','Color','k',...
    'LineStyle','none','fontsize',labelFontSize,'HorizontalAlignment','Center');

% comparision of the norma
dAxes = axes('position',[.08  .12  0.25  0.35]); hold on
annotation('textbox', [.005 .47 .03 .03],...
    'String', 'D','BackgroundColor','none','Color','k',...
    'LineStyle','none','fontsize',labelFontSize,'HorizontalAlignment','Center');

% auto-correlation function
eAxes = axes('position',[.4  .12  .25  .35]);
annotation('textbox', [.33 .47 .03 .03],...
    'String', 'E','BackgroundColor','none','Color','k',...
    'LineStyle','none','fontsize',labelFontSize,'HorizontalAlignment','Center');

% heatmap of feature map
fAxes = axes('position',[.74  .12  .2  .35]); 
annotation('textbox', [.68 .47 .03 .03],...
    'String', 'F','BackgroundColor','none','Color','k',...
    'LineStyle','none','fontsize',labelFontSize,'HorizontalAlignment','Center');

% plot PSP error
pointGap = 20;
xs = (1:pointGap:time_points)'*step;
axes(aAxes)
plot(xs,pspErr(1:pointGap:time_points)/Vnorm,'Color',greys(8,:),'LineWidth',1)
box on
xlabel('$t$','Interpreter','latex','FontSize',labelFontSize)
ylabel('PSP error','FontSize',labelFontSize)
set(gca,'LineWidth',1,'FontSize',gcaFontSize,'Ylim',[0,0.5])

% plot the example trajectory
selYInx = randperm(num_sel,1);
ys = Yt(:,:,selYInx);
% ys = Ytest(:,:,13);
colorsSel = [YlOrRd(6,:);PuRd(8,:);blues(8,:)];
axes(bAxes)

for i = 1:k
    plot(xs,ys(i,1:pointGap:time_points)','Color',colorsSel(i,:),'LineWidth',2)
    hold on
end
hold off
bLgh = legend('y_1','y_2','y_3','Location','northwest');
legend boxoff
set(bLgh,'FontSize',14)
xlabel('$t$','Interpreter','latex','FontSize',labelFontSize)
ylabel('$y_i(t)$','Interpreter','latex','FontSize',labelFontSize)
set(gca,'LineWidth',1,'FontSize',gcaFontSize)


% plot the similarity marix at two time
smSelect = randperm(num_sel,20);
time1 = 1;
time2 = 2e3;
Y1sel = squeeze(Yt(:,time1,smSelect));
Y2sel = squeeze(Yt(:,time2,smSelect));

% order the idex by the cluster
% clusterObj = clustergram(Y1sel,'Cluster','column');
tree = linkage(Y1sel','average');
D = pdist(Y1sel');
leafOrder = optimalleaforder(tree,D);


axes(cAxesi)
imagesc(Y1sel(:,leafOrder)'*Y1sel(:,leafOrder),[-4,4])
colormap(RdBuMap)
set(gca,'XTick','','YTick',[1,10,20])
ylabel('Stimuli','FontSize',labelFontSize)
% annotation('textbox', [.69 .85 .1 .03],...
%     'String', ['t = ', num2str(time1)],'BackgroundColor','none','Color','k',...
%     'LineStyle','none','fontsize',labelFontSize,'VerticalAlignment','middle')

axes(cAxesii)
imagesc(Y2sel(:,leafOrder)'*Y2sel(:,leafOrder),[-4,4])
colormap(RdBuMap)
% annotation('textbox', [.69 .68 .1 .03],...
%     'String', ['t = ', num2str(time2)],'BackgroundColor','none','Color','k',...
%     'LineStyle','none','fontsize',labelFontSize,'VerticalAlignment','middle');
set(gca,'XTick','','YTick',[1,10,20])
c = colorbar;
c.Position = [0.96,0.73,0.01,0.15];
set(gca,'XTick',[1,10,20],'YTick',[1,10,20])
xlabel('Stimuli','FontSize',labelFontSize)
ylabel('Stimuli','FontSize',labelFontSize)

% change of similarity matrix norm, and F'F norm compared with identity
% matrix
axes(dAxes)
plot(xs,SMerror(1:pointGap:time_points),'Color',blues(9,:),'LineWidth',1)
box on
xlabel('$t$','Interpreter','latex','FontSize',labelFontSize)
ylabel('$||\rm{SM_{t} - SM_{0}}||_F/||SM_0||_F$','Interpreter','latex',...
    'FontSize',labelFontSize)
set(gca,'LineWidth',1,'FontSize',16,'YLim',[0,0.1])


% Plot auto correlation function and an exponential fit
[acoef,meanAcf,allTau] = SMhelper.fitAucFun(Yt,step);
% meanAcf = squeeze(mean(mean(acoef,3),2));
% meanTau = fit(xFit,meanAcf,'exp1');
% only plot partial of them
aucSel = randperm(num_sel,10);
axes(eAxes)
% figure
for i = 1:10
    for j = 1:k
        plot((0:length(meanAcf)-1)'*step,acoef(:,j,aucSel(i)),'LineWidth',1,'Color',blues(5,:))
        hold on
    end
end
plot((0:length(meanAcf)-1)'*step,meanAcf,'LineWidth',4,'Color',blues(10,:))
hold off
xlim([0,length(meanAcf)*step])
xlabel('$t$','Interpreter','latex','FontSize',labelFontSize)
ylabel('Auto Corr. Coef.','FontSize',labelFontSize)
set(gca,'FontSize',gcaFontSize,'LineWidth',1)

% Plot the covariance of vectorized feature maps
axes(fAxes)
imagesc(cov(Fvec)); colorbar;
% title('Cov(Ft)')
xlabel('$t$','Interpreter','latex','FontSize',labelFontSize)
ylabel('$t$','Interpreter','latex','FontSize',labelFontSize)
set(gca,'FontSize',gcaFontSize,'LineWidth',1)
c = colorbar;
c.Position = [0.955,0.12,0.01,0.35];

% save the figure

prefix = 'psp_summary_fig2';
% saveas(paperFig2,[sFolder,filesep,prefix,'.fig'])
% print('-depsc',[saveFolder,filesep,prefix,'.eps'])

%% ==================== Figure 3 ==================

% chacterizing the drift in terms of rotational diffusion
% a 2 by 2 figure

paperFig3 = figure; 
set(gcf,'color','w','Units','inches')
pos(3)=9;  %17.4; %8.5;%pos(3)*1.5;
pos(4)=8;%pos(4)*1.5;
set(gcf,'Position',pos)
labelFontSize = 24;
gcaFontSize = 20;

% PSP error
aAxes = axes('position',[.1  .63  .38  .35]); hold on
annotation('textbox', [.01 .98 .03 .03],...
    'String', 'A','BackgroundColor','none','Color','k',...
    'LineStyle','none','fontsize',labelFontSize,'HorizontalAlignment','Center');

% example trajectories
bAxes = axes('position',[.6  .63  .38  .35]);
annotation('textbox', [.52 .98 .03 .03],...
    'String', 'B','BackgroundColor','none','Color','k',...
    'LineStyle','none','fontsize',labelFontSize,'HorizontalAlignment','Center');

% auto-correlation function
cAxes = axes('position',[.1  .12  .38  .35]);
annotation('textbox', [.01 .5 .03 .03],...
    'String', 'C','BackgroundColor','none','Color','k',...
    'LineStyle','none','fontsize',labelFontSize,'HorizontalAlignment','Center');

% heatmap of feature map
dAxes = axes('position',[.6  .12  .38  .35]); 
annotation('textbox', [.52 .5 .03 .03],...
    'String', 'D','BackgroundColor','none','Color','k',...
    'LineStyle','none','fontsize',labelFontSize,'HorizontalAlignment','Center');

% ====================================================
% 3-d scattering showing the representational drift, color indicates time
% ====================================================

exampleSel = randperm(num_sel,1);  % randomly select a stimulus
Yexample = Yt(:,:,exampleSel);
interSep = 5;   % only select data every 10 time points, to eliminate clutter
dotColors = flip(brewermap(size(Yexample,2)/interSep,'Spectral'));


% Also plot a sphere to guid visualization
ridus = sqrt(mean(sum(Yexample.^2,1)));
gridNum = 30;
u = linspace(0, 2 * pi, gridNum);
v = linspace(0, pi, gridNum);


axes(aAxes)
sfh = surf(ridus * cos(u)' * sin(v), ...
    ridus * sin(u)' * sin(v), ...
    ridus * ones(size(u, 2), 1) * cos(v), ...
    'FaceColor', 'w', 'EdgeColor', [.9 .9, .9]);
sfh.FaceAlpha = 0.3000;  % for transparency
hold on
for i = 1:(size(Yexample,2)/interSep)
    plot3(Yexample(1,i*interSep),Yexample(2,i*interSep),Yexample(3,i*interSep),...
        '.','MarkerSize',8,'Color',dotColors(i,:))
end
% grid on
hold off
colormap(dotColors)
c2 = colorbar;
c2.Position = [0.42,0.75,0.015,0.2];

view([45,30])
xlabel('$y_1$','Interpreter','latex','FontSize',labelFontSize)
ylabel('$y_2$','Interpreter','latex','FontSize',labelFontSize)
zlabel('$y_3$','Interpreter','latex','FontSize',labelFontSize)
set(gca,'FontSize',gcaFontSize,'LineWidth',1.5)

% ====================================================
% plot example rotational means quare displacement and a fit of the
% diffusiotn constant
% ====================================================
fitRange = 500;  % only select partial data to fit
aveRMSD = mean(rmsd,1);
logY = log(aveRMSD(1:fitRange)');
logX = [ones(fitRange,1),log((1:fitRange)'*step)];
b = logX\logY;  
Dphi = exp(b(1));  % diffusion constant
exponent = b(2); % factor
pInx = randperm(size(rmsd,1),20);

axes(bAxes)
plot((1:size(rmsd,2))'*step, rmsd(pInx,:)','Color',greys(5,:),'LineWidth',1.5)
hold on
plot((1:size(rmsd,2))*step,mean(rmsd,1),'LineWidth',4,'Color',PuRd(7,:))

% overlap fitted line
yFit = exp(logX*b);
plot((1:fitRange)'*step,yFit,'k--','LineWidth',2)
hold off
xlabel('$\Delta t$','Interpreter','latex','FontSize',labelFontSize)
ylabel('$\langle\varphi^2(\Delta t)\rangle$','Interpreter','latex','FontSize',labelFontSize)
set(gca,'FontSize',20,'LineWidth',1,'XScale','log','YScale','log',...
 'XTick',10.^(1:5),'YTick',10.^(-4:2:0))
set(gca,'FontSize',gcaFontSize,'LineWidth',1.5)
% phiText = texlabel('D_\varphi \propto t^(0.99)');
% thdl = text(1e3,0.2,phiText,'FontSize',20);
% set(thdl,'Rotation',35)

% =========================================================
% Scaling of the diffusion constant and the noise amplitude
% =========================================================
% first, load the data
noiseData = load('./data/noiseAmp0808.mat');
spectrumData = load('./data/eigenSpectr0821.mat');  % new data
% spectrumData = load('./data/eigenSpectr0301.mat');


noiseStd = reshape(noiseData.noiseStd,[40,41]);
noiseAmpl = noiseStd(1,:).^2;
allD_noise = noiseData.allDiffConst(:,1)/4;  % the factor 4 is due to fitting
aveD_noise = mean(reshape(allD_noise,[40,41]),1);
stdD_noise = std(reshape(allD_noise,[40,41]),0,1);
meanExp_noise = mean(noiseData.allDiffConst(:,2));
stdExp_noise = std(noiseData.allDiffConst(:,2));

allEigs = spectrumData.eigens(1:3,:);
sumKeigs = sum(1./(allEigs.^2),1);
allD_spectr = spectrumData.allDiffConst(:,1)/4;
aveExp_spectr = mean(spectrumData.allDiffConst(:,2));
stdExp_spectr = mean(spectrumData.allDiffConst(:,2));

% linear regression to estimate the power law exponent
selInx = 1:41;  % select part of the data to fit
alp_dr = [ones(length(noiseAmpl(selInx)),1),log10(noiseAmpl(selInx)')]\log10(aveD_noise(selInx)');
% alp_dthe =[ones(length(noiseStd),1), log10(noiseStd')]\log10(diffThe_noise(:,1));
b_fit = mean(log10(aveD_noise(selInx)')-log10(noiseAmpl(selInx)'));
% plot noise dependence
X0 = [ones(length(selInx),1),log10(noiseAmpl(selInx)')];


axes(cAxes)
eh1 = errorbar(noiseAmpl(selInx)',aveD_noise(selInx)',stdD_noise(selInx)','o','MarkerSize',8,'MarkerFaceColor',...
    greys(7,:),'Color',greys(7,:),'LineWidth',2,'CapSize',0);
hold on
eh1.YNegativeDelta = []; % only show upper half of the error bar
% lh1 = plot(noiseAmpl(selInx)',10.^(X0*[b_fit;1]),'LineWidth',3,'Color',PuRd(7,:));
% hold off
% this need to be checked in the original data
prefForm = sum(1./noiseData.eigens(1:3).^2)*noiseData.learnRate/2;
plot(cAxes,noiseAmpl(selInx)',noiseAmpl(selInx)'*prefForm,'LineWidth',2,'Color',PuRd(7,:))

% lg = legend(eh1,'$D_{\varphi} \propto \sigma^2 $','Location','northwest');
lg = legend('simulation','theory','Location','northwest');

set(lg,'Interpreter','Latex')
legend boxoff
xlabel('Noise amplitude $(\sigma^2)$','Interpreter','latex','FontSize',labelFontSize)
% ylabel('diffusion constant','FontSize',20)
ylabel('$D_{\varphi}$','Interpreter','latex','FontSize',labelFontSize)
ylim([5e-9,2e-4])
set(gca,'LineWidth',1.5,'FontSize',gcaFontSize,'XScale','log','YScale','log',...
    'YTick',10.^(-8:2:-4))


% ***************************************************
% theory and simulations
% *****************************************************
nBins = 20;
spRange = 10.^[-0.5,2];   % this range depends on the data set used
dbin = (log10(spRange(2)) - log10(spRange(1)))/nBins;
aveSpDs = nan(nBins, 2);  % average and standard deviation
for i = 1:nBins
    inx = log10(sumKeigs) >= log10(spRange(1)) + (i-1)*dbin & log10(sumKeigs) < log10(spRange(1)) + i*dbin;
    aveSpDs(i,:) = [mean(allD_spectr(inx)),std(allD_spectr(inx))];
end
centers = spRange(1)*10.^(dbin*(1:nBins));

% a linear fit based on the averaged diffusion constant
selInx = 1:20;      % remove the last two data points
b_fit_eg = mean(log10(aveSpDs(selInx,1))-log10(centers(selInx)'));
X0_eg = [ones(length(selInx),1),log10(centers(selInx)')];
theoPre = spectrumData.learnRate*spectrumData.noiseStd^2/2;

axes(dAxes)
eh = errorbar(centers',aveSpDs(:,1),aveSpDs(:,2),'o','MarkerSize',8,'MarkerFaceColor',...
    greys(7,:),'Color',greys(7,:),'LineWidth',2,'CapSize',0);
hold on
% plot(centers(selInx)',10.^(X0_eg*[b_fit_eg;1]),'LineWidth',3,'Color',PuRd(7,:));
plot(centers',theoPre*centers','LineWidth',2,'Color',PuRd(7,:))
eh.YNegativeDelta = []; % only show upper half of the error bar
lg = legend('simulation','theory','Location','northwest');

xlabel('$\sum_{i=1}^{k}1/\lambda_i^2$','Interpreter','latex','FontSize',labelFontSize)
ylabel('$D_{\varphi}$','Interpreter','latex','FontSize',labelFontSize)
% set(dAxes,'LineWidth',1.5,'FontSize',gcaFontSize,'XScale','log','YScale','log',...
%     'Ylim',[1e-5,2e-4],'YTick',10.^(-5:1:-2),'XTick',10.^([2,3]))
set(dAxes,'LineWidth',1.5,'FontSize',gcaFontSize,'XScale','log','YScale','log',...
    'Ylim',[1e-6,1e-3],'YTick',10.^(-6:1:-3),'XTick',10.^([0,1,2]))



% save data and figure
% dFile = './data/dataFig3.mat';
% sFig3 = './figures/fig3';
prefix = 'psp_summary_noise_eigSpec_0822';
% dFile = ['./data/',prefix,'.mat'];
% save(dFile,'Yexample','num_sel','rmsd','noiseData','spectrumData')
saveas(paperFig3,[saveFolder,filesep,prefix,'.fig'])
% set(paperFig3,'renderer','Painters','PaperSize',[pos(3) pos(4)])
print('-depsc',[saveFolder,filesep,prefix,'.eps'])
% print('-dpdf',[saveFolder,filesep,prefix,'.pdf'])


%% A closer look at the eigenspectrum and the diffusion constants
nBins = 20;
% spRange = [30,2000];
spRange = 10.^[-0.5,2];
dbin = (log10(spRange(2)) - log10(spRange(1)))/nBins;
aveSpDs = nan(nBins, 2);  % average and standard deviation
aveCenter = nan(nBins,1);
for i = 1:nBins
    inx = log10(sumKeigs) >= log10(spRange(1)) + (i-1)*dbin & log10(sumKeigs) < log10(spRange(1)) + i*dbin;
    aveSpDs(i,:) = [mean(allD_spectr(inx)),std(allD_spectr(inx))];
    aveCenter(i) = mean(log10(sumKeigs(inx)));
end
centers = spRange(1)*10.^(dbin*(1:nBins));

% a linear fit based on the averaged diffusion constant
selInx = 1:20;      % remove the last two data points
b_fit_eg = mean(log10(aveSpDs(selInx,1))-log10(centers(selInx)'));
% plot noise dependence
X0_eg = [ones(length(selInx),1),log10(centers(selInx)')];
% a theoretic line
theoPre = spectrumData.learnRate*spectrumData.noiseStd^2/2;

figHdspD = figure;
set(figHdspD,'Units','inches','Position',[0,0,4,3])
figHdspD = errorbar(centers',aveSpDs(:,1),aveSpDs(:,2),'o','MarkerSize',8,'MarkerFaceColor',...
    greys(7,:),'Color',greys(7,:),'LineWidth',2,'CapSize',0);
hold on
plot(centers(selInx)',10.^(X0_eg*[b_fit_eg;1]),'LineWidth',3,'Color',PuRd(7,:));
plot(centers',theoPre*centers')
figHdspD.YNegativeDelta = []; % only show upper half of the error bar
xlabel('$\sum_i^{k}1/\lambda_i^2$','Interpreter','latex','FontSize',20)
ylabel('$D_{\varphi}$','Interpreter','latex','FontSize',20)
set(gca,'LineWidth',1.5,'FontSize',16,'XScale','log','YScale','log',...
    'YTick',10.^(-5:1:-2),'XTick',10.^([2,3]))
xlim([centers(1),centers(end-3)])

figure
plot(sumKeigs,allD_spectr,'.','MarkerSize',6,'Color', blues(9,:))
xlabel('$\sum_i^{k}1/\lambda_i^2$','Interpreter','latex','FontSize',20)
ylabel('$D_{\varphi}$','Interpreter','latex','FontSize',20)
% set(gca,'LineWidth',1.5,'FontSize',16,'XScale','log','YScale','log',...
%     'XLim',[30,1e6],'YLim',[1e-5,1e-1])
set(gca,'LineWidth',1.5,'FontSize',16,'XScale','log','YScale','log')
hold on

plot(10.^(-1:0.1:2),theoPre*10.^(-1:0.1:2))
plot(10.^aveCenter,theoPre*10.^aveCenter)
plot(10.^aveCenter,aveSpDs(:,1))


%% Change of Frobenius norm, compare with Carl and Andrew's experiment
% select weakly correlatd input by their norm


% order the idex by the cluster
% clusterObj = clustergram(Y1sel,'Cluster','column');
Xtest = X(:,randperm(1e4,1e3));
% Xin = sInx;
Xsel = Xtest;
xtree = linkage(Xsel','average');
xD = pdist(Xsel','cosine');
xleafOrder = optimalleaforder(xtree,xD);

figure
% imagesc(Xtest(:,Xin(xleafOrder))'*Xtest(:,Xin(xleafOrder)),[-1,1])
imagesc(Xtest(:,xleafOrder)'*Xtest(:,xleafOrder),[-1,1])
colorbar

% sInx = Xin(xleafOrder(250:259));
sInx = xleafOrder(480:490);
figure
imagesc(Xtest(:,sInx)'*Xtest(:,sInx))


% Xnorm = sqrt(sum(X.^2,1));
% sInx = find(Xnorm > 0.5 & Xnorm < 0.6);
% sInx = find(Xnorm < 0.5);
% inxSel = randperm(length(sInx),20);
timePoints = [2,50,200,1000,2000,5e3];
fnChg = nan(length(timePoints),50);
for j = 1:50
sInx = randperm(1e2,20);
vec0= Ft(:,:,1)*Xtest(:,sInx);
SM0 = vec0'*vec0;
for i = 1:length(timePoints)
    vec = Ft(:,:,timePoints(i))*Xtest(:,sInx);
    SM = vec'*vec;
    fnChg(i,j) = norm(SM-SM0);
end
end
figure
errorbar(timePoints',mean(fnChg,2),std(fnChg,0,2),'LineWidth',2)
xlabel('$t$','Interpreter','latex','FontSize',24)
ylabel('$||\Delta SM||_F$','Interpreter','latex','FontSize',24)
set(gca,'LineWidth',1.5,'FontSize',20,'XScale','log')


figure
imagesc(SM0)
set(gca,'FontSize',20)
title('$t=1$','Interpreter','latex')
colorbar

figure
imagesc(SM)
set(gca,'FontSize',20)
title('$t=10^4$','Interpreter','latex')
colorbar

% for the weakly samples
fnChgWeak = nan(length(timePoints),1);
vec0= Ft(:,:,1)*Xtest(:,sInx);
SM0 = vec0'*vec0;
for i = 1:length(timePoints)
    vec = Ft(:,:,timePoints(i))*Xtest(:,sInx);
    SM = vec'*vec;
    fnChgWeak(i) = norm(SM-SM0)/norm(SM0);
end
figure
plot(timePoints',fnChgWeak)
xlabel('$t$','Interpreter','latex','FontSize',24)
ylabel('$||\Delta SM||_F$','Interpreter','latex','FontSize',24)
set(gca,'LineWidth',1.5,'FontSize',20,'XScale','log')


Z = (SM-SM0)./SM0;
figure
plot(abs(SM0(:)),abs(Z(:)),'o')
set(gca,'YLim',[0,0.2])
xlabel('$SM$','Interpreter','latex','FontSize',24)
ylabel('$\Delta SM/SM$','Interpreter','latex','FontSize',24)
set(gca,'LineWidth',1.5,'FontSize',20,'XScale','linear')

F0 = Ft(:,:,1);
F1 = Ft(:,:,end);
figure
imagesc(F0)
figure
imagesc(F1)

figure
imagesc(F0'*F0)

figure
imagesc(F1'*F1)

%% PCA of selected example
set1 = brewermap(11,'set1');
figure
for i = 1:length(timePoints)
    vec = Ft(:,:,timePoints(i))*Xtest(:,sInx);
    [COEF, SCORE,~,~,EXPLAINED] = pca(vec');
    subplot(2,3,i)
    hold on
    for j = 1:10
        plot(SCORE(j,1),SCORE(j,2),'o','MarkerSize',10,'MarkerEdgeColor',...
            set1(j,:),'MarkerFaceColor',set1(j,:))
    end
    hold off
    box on
    xlabel(['%',num2str(round(EXPLAINED(1)*100)/100)],'FontSize',16)
    ylabel(['%',num2str(round(EXPLAINED(2)*100)/100)],'FontSize',16)
    title(['t = ',num2str(timePoints(i))])
end