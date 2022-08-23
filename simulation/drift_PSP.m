% function drift_PSP()

% this script simulates the drift in a linear Hebbian/anti-Hebbian network
% performing principal subspace projection task
% the variable names follow Pehlevan and Chklovskii's "why do similarity
% matching objective functions ..."

% the script aslo plots all the figures related to Figure 2 and
% corresponding SI figures in the manuscript

% Notice, we have set learning rate for M and W to be the same,
% corresponding to the case tau = 0.5

type = 'psp';           % 'psp' or 'expansion
tau = 0.5;              %  scaling of the learning rate for M
learnType = 'online';   %  online, offline, batch   

saveFolder = '../figures';

%% learning rate dependent, keep eigen spectrum and noise level, tau the same
input_dim = 10;      
output_dim = 3;

num_sample = 1e4;        % total number of samples generated
tot_iter = 1e5; 

syn_noise_std = 2e-2;    % 1e-2 for offline
learnRate = 5e-2;        % 0.05 for offline

% initilize the synaptic weight matrices
W = randn(output_dim,input_dim);
M = eye(output_dim,output_dim);

input_cov_eigens = [4.5,3.5,1,ones(1,7)*0.01];
assert(length(input_cov_eigens) == input_dim, ...
    'The input convariance matrix does not match input dimension!')

% generate training and testing data
[X, Xtest, ~] = generate_PSP_input(input_cov_eigens,input_dim,output_dim, num_sample);
[V,~] = eigs(X*X',output_dim);
Vnorm = norm(V(:,1:output_dim)*V(:,1:output_dim)','fro');  % used when calculate the normalized PSP error

% ========================================================
% Use offline learning to find the inital solution
% this can speed up the initial stage of simulation
% ========================================================

dt = learnRate;
stdW = syn_noise_std;   
stdM = syn_noise_std;  

% dt0 = 0.01;                % learning rate for the initial phase, can be larger for faster convergence
% tau0 = 0.5;
% PSP_err_no_noise = nan(100,1);   % store the psp error during the pre-noise stage
% 
% for i = 1:2000
%     dt0 = 1/(1e3+i);
%     Y = pinv(M)*W*X; 
%     W = (1-dt0)*W + dt0*Y*X'/num_sample + sqrt(dt)*stdW*randn(output_dim,input_dim);
%     M = (1-dt0)*M + dt0*(Y*Y')/num_sample + sqrt(dt0)*stdM*randn(output_dim,output_dim);
%     F = pinv(M)*W;
% %     disp(norm(F*F'-eye(output_dim),'fro'))
%     disp('psp error:')
%     disp(norm(F'*F - V(:,1:output_dim)*V(:,1:output_dim)','fro')/Vnorm)
% end


% initial stage, show the decrease of PSP error with online learning
tot_inital = 2e3;
step = 1;
time_points = round(tot_inital/step);
pspErr = nan(time_points,1);
for i = 1:tot_inital
        curr_inx = randperm(num_sample,1);  % randm generate one sample
        % generate noise matrices
        xis = randn(output_dim,input_dim);
        zetas = randn(output_dim,output_dim);
        Y = pinv(M)*W*X(:,curr_inx); 
        W = (1-dt)*W + dt*Y*X(:,curr_inx)' + sqrt(dt)*stdW*xis;
        M = (1-dt)*M + dt*Y*Y' +  sqrt(dt)*stdM*zetas; 
  
        if mod(i,step)==0
            temp = pinv(M)*W;  % current feature map
            % PSP error
            if strcmp(type,'psp')
                pspErr(round(i/step)) = norm(temp'*temp - V(:,1:output_dim)*V(:,1:output_dim)','fro')/Vnorm;
            end
        end
end
F = pinv(M)*W;     % initial feature maps

% show the learning curve
f_learnCurve = figure;
pos(3)= 4.5; pos(4)= 3.5;
set(f_learnCurve,'color','w','Units','inches','Position',pos)
% plot(pspErr(1:200),'LineWidth',4)
plot(((1:length(pspErr))*step)',pspErr,'LineWidth',4)
set(gca,'FontSize',20,'LineWidth',1.5)
xlabel('$t$','Interpreter','latex','FontSize',24)
ylabel('PSP error','FontSize',24)
xlim([1 2e2])

% prefix = 'psp_learning_curve';
% saveas(f_learnCurve,[saveFolder,filesep,prefix,'_',date,'_3_','.fig'])
% print('-depsc',[saveFolder,filesep,prefix,'.eps'])

%% now add noise to see how F and output drift

% tot_iter = 5e4;   % total number of updates after entering stationary state
num_sel = 200;      % randomly selected samples used to calculate the drift and diffusion constants
step = 10;          % store every 10 updates
time_points = round(tot_iter/step);
Yt = zeros(output_dim,time_points,num_sel);
Ytest = zeros(output_dim,time_points,size(Xtest,2));     % store the testing stimulus

Ft = zeros(output_dim,input_dim,time_points);        % store the feature maps
sel_inx = randperm(num_sample,num_sel);    % samples indices
Wout = randn(1,output_dim);       % readout vector
readError = nan(tot_iter,1);      % store the average instantaneous readout error

Wout_store = nan(time_points,output_dim);

outEigVecs = nan(output_dim,output_dim,time_points);   % eigenvectors of output

% store all the representations at three time points
subpopulation = 1000;             % number of subpopulation chosen
ensemble_step = 1e4;           
partial_index = randperm(num_sample,subpopulation);   
Y_ensemble = zeros(output_dim,subpopulation,round(tot_iter/ensemble_step) + 1);    %store a subpopulation
pspErr = nan(time_points,1);
featurMapErr = nan(time_points,1);
SMerror = nan(time_points,1);

X0 = F*X(:,sel_inx);
SM0 = X0'*X0;        % similarity matrix at time 0
if strcmp(learnType,'offline')
    for i = 1:tot_iter
        % sample noise matrices
        xis = randn(output_dim,input_dim);
        zetas = randn(output_dim,output_dim);
        Y = pinv(M)*W*X;
        W = (1-dt)*W + dt*Y*X'/tot_iter + sqrt(dt)*stdW*xis;
        M = (1-dt)*M + dt*Y*Y'/tot_iter +  sqrt(dt)*stdM*zetas;

        if mod(i,step)==0
            
            % PSP error
            if strcmp(type,'psp')
                pspErr(round(i/step)) = norm(temp'*temp - V(:,1:output_dim)*V(:,1:output_dim)','fro')/Vnorm;
            end
            featurMapErr(round(i/step)) = norm(temp*temp' - eye(output_dim),'fro');
            Ysel = temp*X(:,sel_inx);
            SMerror(round(i/step)) = norm(Ysel'*Ysel - SM0,'fro')/(norm(SM0,'fro') + 1e-10);
            % representation of test stimuli
            Ytest(:,round(i/step),:) = temp*Xtest;
                 
        end
        
        
        % store population representations every 10000 steps
        if mod(i, ensemble_step)==0 || i == 1
           Y_ensemble(:,:,round(i/1e4)+1) = pinv(M)*W*X(:,partial_index);
        end
    
    end
elseif strcmp(learnType,'online')
    for i = 1:tot_iter
        curr_inx = randperm(num_sample,1);  % randm generate one sample
        % generate noise matrices
        xis = randn(output_dim,input_dim);
        zetas = randn(output_dim,output_dim);
        Y = pinv(M)*W*X(:,curr_inx); 
        W = (1-dt)*W + dt*Y*X(:,curr_inx)' + sqrt(dt)*stdW*xis;
        M = (1-dt)*M + dt*Y*Y' +  sqrt(dt)*stdM*zetas; 
  
        if mod(i,step)==0
            temp = pinv(M)*W;  % current feature map
            Yt(:,round(i/step),:) = temp*X(:,sel_inx);
            Ft(:,:,round(i/step)) = temp; % store
                
            % PSP error
            if strcmp(type,'psp')
                pspErr(round(i/step)) = norm(temp'*temp - V(:,1:output_dim)*V(:,1:output_dim)','fro')/Vnorm;
            end
            featurMapErr(round(i/step)) = norm(temp*temp' - eye(output_dim),'fro');
            Ysel = temp*X(:,sel_inx);
            SMerror(round(i/step)) = norm(Ysel'*Ysel - SM0,'fro')/(norm(SM0,'fro') + 1e-10);
            % representation of test stimuli
            Ytest(:,round(i/step),:) = temp*Xtest;
                    
        end
        
        % store population representations every 10000 steps
        if mod(i, ensemble_step)==0 || i == 1
           Y_ensemble(:,:,round(i/1e4)+1) = pinv(M)*W*X(:,partial_index);
        end     
    end
end


%% Analysis of the esembles of neural activity
Fvec = reshape(Ft,output_dim*input_dim,time_points);

% plot the ensemble representations
figYens = figure;
set(figYens,'Units','inches','Position',[0,0,4,3])
ensem_sel = [8, 11];   % this should depdend on the learning type
for i = 1:length(ensem_sel)
    scatterHd = plot3(Y_ensemble(1,:,ensem_sel(i)),Y_ensemble(2,:,ensem_sel(i)),...
        Y_ensemble(3,:,ensem_sel(i)),'.','MarkerSize',6);
    hold on
    scatterHd.MarkerFaceColor(4) = 0.2;
    scatterHd.MarkerEdgeColor(4) = 0.2;
    grid on
end
% lg = legend('$t = 0$','$t= 5\times 10^4$','interpreter','latex')
xlabel('$y_1$','Interpreter','latex','FontSize',20)
ylabel('$y_2$','Interpreter','latex','FontSize',20)
zlabel('$y_3$','Interpreter','latex','FontSize',20)
set(gca,'FontSize',16,'XLim',[-4,4],'YLim',[-4,4],'ZLim',[-3,3])

% save the figure
% prefix = [type,'_ensemble_orientation_dt3e3_',date,'_3'];
% saveas(figYens,[saveFolder,filesep,prefix,'.fig'])
% print('-depsc',[saveFolder,filesep,prefix,'.eps'])

rmsd = SMhelper.rotationalMSD(Yt);
[dph,ephi] = SMhelper.fitRotationDiff(rmsd,step,2000,'mean_log');
[dph_linear,~] = SMhelper.fitRotationDiff(rmsd,step,2000,'mean_linear');

% prefix = 'oja_readout_fig2';
% saveas(hbh,[saveFolder,filesep,prefix,'.fig'])
% print('-depsc',[saveFolder,filesep,prefix,'.eps'])

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
sFolder = '../figures';
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
plot(xs,pspErr(1:pointGap:time_points),'Color',greys(8,:),'LineWidth',1)
box on
xlabel('$t$','Interpreter','latex','FontSize',labelFontSize)
ylabel('PSP error','FontSize',labelFontSize)
set(aAxes,'LineWidth',1,'FontSize',gcaFontSize,'Ylim',[0,1])

% plot the example trajectory
selYInx = randperm(num_sel,1);
ys = Yt(:,:,selYInx);
% ys = Ytest(:,:,13);
colorsSel = [YlOrRd(6,:);PuRd(8,:);blues(8,:)];
axes(bAxes)

for i = 1:output_dim
    plot(xs,ys(i,1:pointGap:time_points)','Color',colorsSel(i,:),'LineWidth',2)
    hold on
end
hold off
bLgh = legend('y_1','y_2','y_3','Location','northwest');
legend boxoff
set(bLgh,'FontSize',14)
xlabel('$t$','Interpreter','latex','FontSize',labelFontSize)
ylabel('$y_i(t)$','Interpreter','latex','FontSize',labelFontSize)
set(bAxes,'LineWidth',1,'FontSize',gcaFontSize)


% plot the similarity marix at two time points
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
imagesc(Y1sel(:,leafOrder)'*Y1sel(:,leafOrder),[-6,6])
colormap(RdBuMap)
set(gca,'XTick','','YTick',[1,10,20])
ylabel('Stimuli','FontSize',labelFontSize)

axes(cAxesii)
imagesc(Y2sel(:,leafOrder)'*Y2sel(:,leafOrder),[-6,6])
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
set(dAxes,'LineWidth',1,'FontSize',gcaFontSize,'YLim',[0,0.5])


% Plot auto correlation function and an exponential fit
[acoef,meanAcf,allTau] = SMhelper.fitAucFun(Yt,step);
% only plot partial of them
aucSel = randperm(num_sel,10);
axes(eAxes)
% figure
for i = 1:10
    for j = 1:output_dim
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

% prefix = 'psp_summary_fig2';
% saveas(paperFig2,[sFolder,filesep,prefix,'_',date,'_2','.fig'])
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

exampleSel = randperm(num_sel,1);  % randomly select a 
% exampleSel = 110;
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

% enhance visualization
radius_new = ridus;   % this should be modified based on the effect
set(aAxes,'XLim',[-radius_new,radius_new],'YLim',[-radius_new,radius_new],...
    'ZLim',[-radius_new,radius_new])

view([45,30])
xlabel('$y_1$','Interpreter','latex','FontSize',labelFontSize)
ylabel('$y_2$','Interpreter','latex','FontSize',labelFontSize)
zlabel('$y_3$','Interpreter','latex','FontSize',labelFontSize)
set(gca,'FontSize',gcaFontSize,'LineWidth',1.5)

% =================================================================
% plot example rotational means quare displacement and a fit of the
% diffusiotn constant
% =================================================================
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
% plot((1:size(rmsd,2))*step,mean(rmsd,1),'LineWidth',4,'Color',PuRd(7,:))

% overlap fitted line
yFit = exp(logX*b);
plot((1:fitRange)'*step,yFit,'k--','LineWidth',2)
hold off
xlabel('$\Delta t$','Interpreter','latex','FontSize',labelFontSize)
ylabel('$\langle\varphi^2(\Delta t)\rangle$','Interpreter','latex','FontSize',labelFontSize)
set(gca,'FontSize',20,'LineWidth',1,'XScale','log','YScale','log',...
 'XTick',10.^(1:5),'YTick',10.^(-4:2:2))
set(gca,'FontSize',gcaFontSize,'LineWidth',1.5)
% phiText = texlabel('D_\varphi \propto t^(0.99)');
% thdl = text(1e3,0.2,phiText,'FontSize',20);
% set(thdl,'Rotation',35)

% =========================================================
% Scaling of the diffusion constant and the noise amplitude
% =========================================================
% first, load the data
noiseData = load('../data/noiseAmp0808.mat');
% spectrumData = load('../data/eigenSpectr0821.mat');  % new data
spectrumData = load('../data/eigenSpectr_08222022.mat');
% spectrumData = load('./data/eigenSpectr0301.mat');


syn_noise_std = reshape(noiseData.noiseStd,[40,41]);
noiseAmpl = syn_noise_std(1,:).^2;
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
theoPre = spectrumData.learnRate*spectrumData.noiseStd^2/8;

axes(dAxes)
eh = errorbar(centers',aveSpDs(:,1),aveSpDs(:,2),'o','MarkerSize',8,'MarkerFaceColor',...
    greys(7,:),'Color',greys(7,:),'LineWidth',2,'CapSize',0);
hold on
% plot(centers(selInx)',10.^(X0_eg*[b_fit_eg;1]),'LineWidth',3,'Color',PuRd(7,:));
plot(centers',theoPre*centers','LineWidth',2,'Color',PuRd(7,:))
eh.YNegativeDelta = []; % only show upper half of the error bar
lg = legend('simulation','theory','Location','northwest');
hold off
xlabel('$\sum_{i=1}^{k}1/\lambda_i^2$','Interpreter','latex','FontSize',labelFontSize)
ylabel('$D_{\varphi}$','Interpreter','latex','FontSize',labelFontSize)
% set(dAxes,'LineWidth',1.5,'FontSize',gcaFontSize,'XScale','log','YScale','log',...
%     'Ylim',[1e-5,2e-4],'YTick',10.^(-5:1:-2),'XTick',10.^([2,3]))
set(dAxes,'LineWidth',1.5,'FontSize',gcaFontSize,'XScale','log','YScale','log',...
    'Ylim',[1e-7,1e-4],'YTick',10.^(-7:1:-4),'XTick',10.^([-1,0,1,2]))

prefix = 'psp_fit_rotational_fig2_2';
saveas(paperFig3,[sFolder,filesep,prefix,'_',date,'_3','.fig'])
print('-depsc',[saveFolder,filesep,prefix,'.eps'])


%% A closer look at the eigenspectrum and the diffusion constants
%{
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
%}