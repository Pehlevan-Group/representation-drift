% using non-negative similarity matching to explore the drift
% representation in piriform cortex

close all
clear
%% model parameters
param.dimIn =  50;     % input dim
param.dimOut = 5e2;     % output dim
param.odorSp = 0.2;     % sparsity of odor stimulit
param.fwdSp = 0.1;      % sparsity of forward matrix

param.odoStat = 'gauss';  % distribution of odor concentration, gaussan, or logmormal
param.bgStat = 'uniform'; % back ground odor statistics
param.numOdor = 1e3;        % number of background odor sampled
param.tgNum = 10;         % number of target odors

% generate background odor stimuli and target odor
concMask = rand(param.dimIn,param.numOdor) < param.odorSp;
bgOdors = rand(param.dimIn,param.numOdor).* concMask;

concMask = rand(param.dimIn,param.tgNum) < param.odorSp;
tgOdors = 0.5*ones(param.dimIn,param.tgNum).* concMask;
% parameters for learning 
noiseStd = 1e-2; % 0.005 for ring
learnRate = 0.1;

nonzeroInx = rand(param.dimOut,param.dimIn) < param.fwdSp; 
param.W = rand(param.dimOut,param.dimIn).*nonzeroInx/param.dimIn/param.fwdSp;   % initialize the forward matrix
param.M = eye(param.dimOut);   % lateral connection if using simple nsm
param.lbd1 = 0.01;     % 0.05 regularization for the simple nsm
param.lbd2 = 0.01;     % 0.01


param.alpha = 0.1;  % 80 for regular, 150 for weakly input
param.beta = 2; 
param.gy = 0.1;   % update step for y
% param.gz = 0.1;   % update step for z
% param.gv = 0.2;   % update step for V
param.b = zeros(param.dimOut,1);  % biase
param.learnRate = learnRate;  % learning rate for W and b
param.noise =  noiseStd;   % stanard deivation of noise 

BatchSize = 1;    % minibatch used to to learn
learnType = 'snsm';  % snsm, batch, online

% input similarity
SM = bgOdors'*bgOdors;
% figure
% imagesc(SM)
% colorbar
%% using non-negative similarity matching to update matrix
% generate input from grid filds

tot_iter = 2e3;   % total interation
testInver = 1e2;  % test interval
tgResp = cell(round(tot_iter/testInver),1);


if strcmp(learnType, 'online')
    for i = 1:1000
        x = X(:,randperm(t,1));  % randomly select one input
        [states, param] = PlaceCellhelper.neuralDynOnline(x,Y0,Z0,V, param);
        y = states.y;
        % update weight matrix
        param.W = (1-param.learnRate)*param.W + y*x';
        param.b = (1-param.learnRate)*param.b + param.learnRatessqrt(param.alpha)*y;
        V = states.V;   % current 
    end
elseif strcmp(learnType, 'batch')
    for i = 1:tot_iter
        x = gdInput(:,randperm(param.ps*param.ps,BatchSize));
%         x = X(:,randperm(param.ps*param.ps,BatchSize));  % randomly select one input
        [states, param] = PlaceCellhelper.neuralDynBatch(x,Y0,Z0,V, param);
        y = states.Y;
        % update weight matrix
        param.W = (1-param.learnRate)*param.W + param.learnRate*y*x'/BatchSize;
        param.b = (1-param.learnRate)*param.b + param.learnRate*sqrt(param.alpha)*mean(y,2);
        V = states.V;   % current 
    end
% simple non-negative similarity matching
elseif strcmp(learnType, 'snsm')
    ystart = zeros(param.dimOut,BatchSize);  % inital of the output
    ystest = zeros(param.dimOut,param.tgNum);
    
    % generate position code by the grid cells    
    for i = 1:tot_iter
        odorInput = bgOdors(:,randperm(param.numOdor,BatchSize));
        states= Piriformhelper.nsmDynBatch(odorInput,ystart, param);
        y = states.Y;
        
        % update weight matrix
%         param.W = (1-param.learnRate)*param.W + param.learnRate*y*positions'/BatchSize;
%         param.M = (1-param.learnRate)*param.M + param.learnRate*y*y'/BatchSize;
        param.M = (1-param.learnRate)*param.M + param.learnRate*y*y'/BatchSize + ...
            sqrt(param.learnRate)*param.noise*randn(param.dimOut,param.dimOut);
        param.b = (1-param.learnRate)*param.b + param.learnRate*sqrt(param.alpha)*mean(y,2);
%         V = states.V;   % current 
        if mod(i,testInver)==1
            states= Piriformhelper.nsmDynBatch(tgOdors,ystest, param);
            tgResp{1+round(i/testInver)} = states.Y;
        end
    end
end



%% Analysis and plot
% change of similarity
PearsonCorrs = nan(length(tgResp),param.tgNum);
for i = 1:length(tgResp)
    for j = 1:param.tgNum
        C = corrcoef(tgResp{i}(:,j),tgResp{1}(:,j));
        PearsonCorrs(i,j) = C(1,2);
    end
end

% continous with target
testOdor = tgOdors(:, randperm(param.tgNum,1));
tgLen = 10;  % target odor presentation
bgLen = 5;   % background odor
sesssions = 20;
pvs = nan(param.dimOut, sesssions);
for i = 1:sesssions
    ystart = zeros(param.dimOut,1);  % initial condition
    for j = 1:tgLen
        states= Piriformhelper.nsmDynBatch(testOdor,ystart, param);
        y = states.Y;
        param.M = (1-param.learnRate)*param.M + param.learnRate*y*y'/BatchSize + ...
            sqrt(param.learnRate)*param.noise*randn(param.dimOut,param.dimOut);
        param.b = (1-param.learnRate)*param.b + param.learnRate*sqrt(param.alpha)*mean(y,2);
    end
    
    % test
    pvs(:,i) = y;
    
    % background odor
    for j = 1:bgLen
        states= Piriformhelper.nsmDynBatch(bgOdors(:,randperm(param.numOdor,1)),ystart, param);
        y = states.Y;
        param.M = (1-param.learnRate)*param.M + param.learnRate*y*y'/BatchSize + ...
            sqrt(param.learnRate)*param.noise*randn(param.dimOut,param.dimOut);
        param.b = (1-param.learnRate)*param.b + param.learnRate*sqrt(param.alpha)*mean(y,2);
        
    end
    
end

figure
imagesc(cov)

%% 
% y = allY(:,:,end);
% =============== Place field  of all the place cells ===============
pfs = y*posiGram'./(sum(y,2)*ones(1,param.ps*param.ps));
figure
hp = tight_subplot(10,10);
sel_inx = randperm(param.Np, min(100,param.Np));
for i = 1:length(sel_inx)
%     axes(hp(i))
%     subplot(10,10,i)
    pf = y(sel_inx(i),:);
%     pf = pfs(sel_inx(i),:);
    imagesc(hp(i),reshape(pf,param.ps,param.ps));
    hp(i).XAxis.Visible = 'off';
    hp(i).YAxis.Visible = 'off';
end

% suface plot of an exmaple place field
PMat = reshape(y(randperm(param.Np,1),:),param.ps,param.ps);
figure
surf(PMat)

% estimate the peak positions of the place field
[~, pkInx] = sort(y,2,'descend');

pkMat = zeros(param.ps,param.ps);
pkMat(pkInx(:,1)) = 1;

figure
imagesc(pkMat)

% amplitude of place fields
z = max(y,[],2);
figure
histogram(z(z>0))
xlim([2,6])
xlabel('Amplitude','FontSize',24)
ylabel('count','FontSize',24)
set(gca,'LineWidth',1.5, 'FontSize',24)


% ================= visualize the input ===========================
% input examples
figure
hinput = tight_subplot(5,5);
sel_inx = randperm(param.ps*param.ps, 25);
for i = 1:length(sel_inx)
    imagesc(hinput(i),reshape(gdInput(:,sel_inx(i)),24,25))
    hinput(i).XAxis.Visible = 'off';
    hinput(i).YAxis.Visible = 'off';
end

% input similarity
figure
imagesc(gdInput'*gdInput,[70,110])
colorbar
xlabel('position index','FontSize',24)
ylabel('position index','FontSize',24)
set(gca,'LineWidth',1.5, 'FontSize',20)
title('Input similarity matrix','FontSize',24)

figure
imagesc(gdInput(:,1:200))
colorbar
xlabel('position index (partial)','FontSize',24)
ylabel('grid cell index','FontSize',24)
set(gca,'LineWidth',1, 'FontSize',20)
title('Input','FontSize',24)


% output matrix and similarity matrix
figure
imagesc(y)
colorbar
xlabel('position index ','FontSize',24)
ylabel('place cell index','FontSize',24)
set(gca,'LineWidth',1, 'FontSize',20)
title('Output','FontSize',24)


figure
imagesc(y'*y)
xlabel('position index ','FontSize',24)
ylabel('position index ','FontSize',24)
set(gca,'LineWidth',1, 'FontSize',20)
title('Output Similarity','FontSize',24)



% ============== visualize the learned matrix =======================
% feedforward connection
figure
imagesc(param.W,[0,0.03])
colorbar
xlabel('grid cell','FontSize',24)
ylabel('place cell','FontSize',24)
set(gca,'LineWidth',1.5,'FontSize',24)

% histogram of W
figure
histogram(param.W(param.W<0.05))
xlabel('$W_{ij}$','Interpreter','latex','FontSize',24)
ylabel('Count','FontSize',24)
set(gca,'LineWidth',1.5,'FontSize',24)

% in term of individual palce cells
figure
plSel = randperm(param.Np,15);
pwhdl = tight_subplot(3,5);
for i = 1:15
    imagesc(pwhdl(i),reshape(param.W(plSel(i),:),24,25))
    pwhdl(i).XAxis.Visible = 'off';
    pwhdl(i).YAxis.Visible = 'off'; 
end


% ============== the feature map: GA =========================
GA = gridFields*param.W';
% GA = gridFields*randn(param.Ng,param.Np)*0.01;
figure
gaHdl = tight_subplot(10,10);
sel_inx = randperm(param.Np, min(100,param.Np));
for i = 1:length(sel_inx)
    imagesc(gaHdl(i),reshape(GA(:,sel_inx(i)),param.ps,param.ps));
    gaHdl(i).XAxis.Visible = 'off';
    gaHdl(i).YAxis.Visible = 'off';
end

% 
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
imagesc(Mhat,[0,0.1])
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

%% Continuous Nosiy update

tot_iter = 2e3;
num_sel = 200;
step = 2;
time_points = round(tot_iter/step);

pks = nan(param.Np,time_points);    % store all the peak positions
pkAmp = nan(param.Np,time_points);  % store the peak amplitude
placeFlag = nan(param.Np,time_points); % determine whether a place field
% store the weight matrices to see if they are stable
timeGap = 50;
allW = nan(param.Np,param.Ng,round(tot_iter/timeGap));
allM = nan(param.Np,param.Np,round(tot_iter/timeGap));
allY = nan(param.Np,param.ps^2,round(tot_iter/timeGap)); % store all the population vectors

ampThd = 0.01;   % amplitude threshold
% testing data, only used when check the representations
% Xsel = gdInput(:,1:10:end);
Y0 = zeros(param.Np,size(gdInput,2));
Z0 = zeros(numIN,size(gdInput,2));
% Yt = nan(param.Np,size(param.Np,2),time_points);
if strcmp(learnType, 'online')
    for i = 1:tot_iter
        x = X(:,randperm(t,1));  % randomly select one input
        [states, params] = MantHelper.neuralDynOnline(x,Y0,Z0,V, params);
        y = states.y;
        % update weight matrix
        params.W = (1-params.learnRate)*params.W + params.learnRate*y*x' + ...
            sqrt(params.learnRate)*params.noise*randn(k,2);
        params.b = (1-params.learnRate)*params.b + params.learnRate*sqrt(params.alpha)*y;
        V = states.V;   % current feedback matrix
        
    end
elseif strcmp(learnType, 'batch')
    y0 = zeros(param.Np, BatchSize);  % initialize mini-batch
    z0 = zeros(numIN,BatchSize);  % initialize interneurons

    for i = 1:tot_iter
        x = gdInput(:,randperm(param.ps*param.ps,BatchSize));
%         x = X(:,randperm(param.ps*param.ps,BatchSize));  % randomly select one input
        [states, param] = PlaceCellhelper.neuralDynBatch(x,y0,z0,V, param);
        y = states.Y;
        % update weight matrix
        param.W = (1-param.learnRate)*param.W + param.learnRate*y*x'/BatchSize...
            + sqrt(param.learnRate)*param.noise*randn(param.Np,param.Ng);
        param.b = (1-param.learnRate)*param.b + param.learnRate*sqrt(param.alpha)*mean(y,2);
        V = states.V;   % current 
        
        if mod(i, step) == 0
            [states_fixed,param] = PlaceCellhelper.neuralDynBatch(gdInput,Y0,Z0,V, param);
            [~,pkInx] = sort(states_fixed.Y,2, 'descend');
            pkAmp(:,round(i/step)) = states_fixed.Y((pkInx(:,1)-1)*param.Np + (1:param.Np)');
            pcInx = pkAmp(:,round(i/step)) > ampThd;  % find the place cells
            pks(pcInx,round(i/step)) = pkInx(pcInx,1);
%             Yt(:,:,round(i/step)) = states_fixed.Y;

        end
    end
elseif strcmp(learnType, 'snsm')
    
    ystart = zeros(param.Np,BatchSize);  % inital of the output
    
    % generate position code by the grid cells    
    for i = 1:tot_iter
        positions = gdInput(:,randperm(param.ps*param.ps,BatchSize));
        states = PlaceCellhelper.nsmDynBatch(positions,ystart, param);
        y = states.Y;
        
        % noisy update weight matrix
        param.W = (1-param.learnRate)*param.W + param.learnRate*y*positions'/BatchSize + ...
            sqrt(param.learnRate)*param.noise*randn(param.Np,param.Ng);  % 9/16/2020
%         param.M = (1-param.learnRate)*param.M + param.learnRate*y*y'/BatchSize + ...
%             sqrt(param.learnRate)*param.noise*randn(param.Np,param.Np);
        param.M = (1-param.learnRate)*param.M + param.learnRate*y*y'/BatchSize;
        param.b = (1-param.learnRate)*param.b + param.learnRate*sqrt(param.alpha)*mean(y,2);
%         param.b = (1-param.learnRate)*param.b + param.learnRate*sqrt(param.alpha)*mean(y,2) + ...
%             sqrt(param.learnRate)*param.noise*randn(param.Np,1);
        
        % store and check representations
        Y0 = zeros(param.Np,size(gdInput,2));
        if mod(i, step) == 0
            states_fixed = PlaceCellhelper.nsmDynBatch(gdInput,Y0, param);
            [~,pkInx] = sort(states_fixed.Y,2, 'descend');
            pkAmp(:,round(i/step)) = states_fixed.Y((pkInx(:,1)-1)*param.Np + (1:param.Np)');
            pcInx = pkAmp(:,round(i/step)) > ampThd;  % find the place cells
            pks(pcInx,round(i/step)) = pkInx(pcInx,1);
%             Yt(:,:,round(i/step)) = states_fixed.Y;
        end
        
        if mod(i,timeGap) ==0
            allW(:,:,round(i/timeGap)) = param.W;
            allM(:,:,round(i/timeGap)) = param.M;
            allY(:,:,round(i/timeGap)) = states_fixed.Y;
        end
       
    end
end

%% analyzing the drift behavior
% pca of the peak amplitude=  dynamcis
[COEFF, SCORE, ~, ~, EXPLAINED] = pca(pkAmp');


figure
plot(cumsum(EXPLAINED),'LineWidth',3)
xlabel('pc','FontSize',24)
ylabel('cummulative variance','FontSize',24)
set(gca,'FontSize',20,'LineWidth',1.5)

%% Peak dynamics
% shift of peak positions
epInx = randperm(param.Np,1);  % randomly slect
epPosi = [floor(pks(epInx,:)/param.ps);mod(pks(epInx,:),param.ps)]+randn(2,size(pks,2))*0.1;

specColors = brewermap(size(epPosi,2), 'Spectral');

% colors indicate time
figure
hold on
for i=1:size(pks,2)
    plot(epPosi(1,i),epPosi(2,i),'^','MarkerSize',4,'MarkerEdgeColor',...
        specColors(i,:),'LineWidth',1.5)
end
hold off
box on
xlabel('x position','FontSize',24)
ylabel('y position','FontSize',24)
set(gca,'FontSize',24,'LineWidth',1.5)

% 2d trajectory
figure
plot(epPosi(1,:),epPosi(2,:), 'o-','MarkerSize',4,'LineWidth',2)
xlabel('x position','FontSize',24)
ylabel('y position','FontSize',24)
set(gca,'FontSize',24,'LineWidth',1.5)

% distribution of step size
ixs = epPosi(1,:)>0;
times =1:size(epPosi,2);
pkTimePoints = times(ixs);
silentInt = diff(pkTimePoints)-1; % silent interval
figure
histogram(silentInt(silentInt > 0)*step)
xlabel('$\Delta t$','Interpreter','latex','FontSize',24)
ylabel('count','FontSize',24)
set(gca,'FontSize',24,'LineWidth',1.5)


temp = epPosi(:,ixs);
dp = [diff(temp(1,:));diff(temp(2,:))];
ds = sqrt(sum(dp.^2,1));

% distribution of
figure
histogram(log(ds(ds>=1)),32)
xlabel('$\ln(\Delta d)$','Interpreter','latex','FontSize',24)
ylabel('count','FontSize',24)
set(gca,'FontSize',24,'LineWidth',1.5)

[f,X] = ecdf(ds(ds>=1 & ds < 20));
figure
plot(X,1 - f,'LineWidth',3)
xlabel('$\log_{10}(\Delta d)$','Interpreter','latex','FontSize',24)
ylabel('$1-F(x)$','Interpreter','latex','FontSize',24)
set(gca,'XScale','log','YScale','log','LineWidth',1.5,'FontSize',24)


% statistics of jump and local diffusion
ds = diff(epPosi,1,2);
dsTraj = sqrt(sum(ds.^2,1))/param.ps;
figure
plot(dsTraj,'LineWidth',1)
xlabel('time','FontSize',24)
ylabel('$\Delta d$','Interpreter','latex','FontSize',24)
set(gca,'LineWidth',1.5,'FontSize',20)


%% Peak amplitdues
% trajectory of peak
figure
plot((1:size(pkAmp,2))*step,pkAmp(epInx,:),'LineWidth',2)
xlabel('Time','FontSize',24)
ylabel('Peak amplitute','FontSize',24)
set(gca,'FontSize',24,'LineWidth',1.5)

% distribution of amplitude
figure
histogram(pkAmp(epInx,:))
xlabel('Peak amplitude','FontSize',24)
ylabel('Count','FontSize',24)
set(gca,'FontSize',24,'LineWidth',1.5)

% heatmap of peaks
figure
imagesc(pkAmp,[0,5])
colorbar

figure
imagesc(pkAmp'*pkAmp)

% fraction of cells that are place cells
figure
plot((1:size(pkAmp,2))*step,mean(pkAmp > 0.1,1))
xlabel('$t$','Interpreter','latex')
ylabel('fraction of place cell')

% peak amplitude of single place cell
pcSel = randperm(param.Np,1); %randomly slect one place cell
figure
plot((1:size(pkAmp,2))*step,pkAmp(pcSel,:))
xlabel('$t$','Interpreter','latex')
ylabel('Peak amplitude')

% peak position evolve with time
tps = [1,200,400];  % select several time point to plot
figure
for i = 1:length(tps)
    pkMat = zeros(param.ps,param.ps);
    pkMat(pks(:,tps(i))) = 1;
    subplot(1,length(tps),i)
    imagesc(pkMat)
    title(['iteration: ',num2str(step*tps(i))],'FontSize',20)
    xlabel('x position','FontSize',20)
    ylabel('y position','FontSize',20)
    set(gca,'FontSize',20,'LineWidth',1)
end

% calculate the overall peak position movement
xs = floor(pks/param.ps);
ys = mod(pks,param.ps);
dxs = xs - xs(:,1);
dys = ys - ys(:,1);
shiftEcul = sqrt(dxs.^2 + dys.^2)/param.ps;  % in unit of m

figure
histogram(shiftEcul(:,end))
xlabel('peak shift $|\Delta r|$', 'Interpreter','latex','FontSize',24)
ylabel('count','FontSize',24)
set(gca,'LineWidth',1,'FontSize',20)
% change of the weight matrix, W
% figure
% for i = 1:size(allW,3)
%     subplot(1,size(allW,3),i)
%     imagesc(allW(:,:,i),[-0.02,0.1])
% end
% 
% % change of the weight matrix, M
% figure
% for i = 1:size(allM,3)
%     M = allM(:,:,i) - diag(diag(allM(:,:,i)));
%     subplot(1,size(allM,3),i)
%     imagesc(M,[-0.02,0.05])
% end
% 
% figure
% histogram(M(:))
% 
% 
% figure
% W = allW(:,:,1);
% histogram(W(:))

% changes of matrix norm
chgW = nan(size(allW,3)-1,1);
chgM = nan(size(allM,3)-1,1);
for i = 1:size(allW,3)-1
    chgW(i) = norm(allW(:,:,i+1) - allW(:,:,1))/norm(allW(:,:,1));
    chgM(i) = norm(allM(:,:,i+1) - allM(:,:,1))/norm(allM(:,:,1));
end


% relative change of W
figure
plot((1:size(allW,3)-1)'*timeGap,chgW,'LineWidth',2)
xlabel('iteration','FontSize',20)
ylabel('$$\frac{||W_t - W_0||_F^2}{||W_0||_F^2}$$','Interpreter','latex','FontSize',20)
set(gca,'FontSize',20,'LineWidth',1,'YLim',[0,0.4])

% relative change of M
figure
plot((1:size(allM,3)-1)'*timeGap,chgM,'LineWidth',2)
xlabel('iteration','FontSize',20)
ylabel('$$\frac{||M_t - M_0||_F^2}{||M_0||_F^2}$$','Interpreter','latex','FontSize',20)
set(gca,'FontSize',20,'LineWidth',1,'YLim',[0,0.7])

ws1 = squeeze(allW(10,33,:));
ws2 = squeeze(allW(39,87,:));

figure
plot(ws1)
hold on
plot(ws2)
hold off

figure
subplot(1,2,1)
imagesc(allW(:,:,1))
colorbar
subplot(1,2,2)
imagesc(allW(:,:,40))
colorbar

figure
imagesc(param.W)

% in term of individual palce cells
figure
plSel = randperm(param.Np,15);
pwhdl = tight_subplot(3,5);
for i = 1:15
    imagesc(pwhdl(i),reshape(param.W(plSel(i),:),24,25))
    pwhdl(i).XAxis.Visible = 'off';
    pwhdl(i).YAxis.Visible = 'off'; 
end


ms1 = squeeze(allM(10,33,:));
ms2 = squeeze(allM(39,87,:));

figure
plot(ms1)
hold on
plot(ms2)
hold off

figure
scatter(ms1,ms2)