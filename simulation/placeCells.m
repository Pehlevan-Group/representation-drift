% using non-negative similarity matching to learn a 2-d place cells based
% on predefined grid cells patterns. Based on the paper lian et al 2020
% Model 1m x 1m 2d environemnt, number of modules of grid cells is 4
clear
close all
%% model parameters
param.ps =  32;     % number of positions along each dimension
param.Nlbd = 5;     % number of different scales/spacing
param.Nthe = 6;     % number of rotations
param.Nx =  3;      % offset of x-direction
param.Ny = 3;       % offset of y-direction
param.Ng = param.Nlbd*param.Nthe*param.Nx*param.Ny;   % total number of grid cells
param.Np = 200;   % number of place cells, default 20*20

param.baseLbd = 0.2;   % spacing of smallest grid RF, default 0.28
param.sf =  1.42;       % scaling factor between adjacent module

% parameters for learning 
noiseStd =0.005;          % 0.01 for 2d, 5 grid mode
learnRate = 0.05;     % default 0.05

param.W = 0.5*randn(param.Np,param.Ng);   % initialize the forward matrix
param.M = eye(param.Np);        % lateral connection if using simple nsm
param.lbd1 = 0.001;               % 0.15 for 400 place cells and 5 modes
param.lbd2 = 0.02;              % 0.05 for 400 place cells and 5 modes


param.alpha = 35;  % 85 for regular,95 for 5 grid modes, 150 for weakly input
param.beta = 2; 
param.gy = 0.05;   % update step for y
param.gz = 0.1;   % update step for z
param.gv = 0.2;   % update step for V
param.b = zeros(param.Np,1);  % biase
param.learnRate = learnRate;  % learning rate for W and b
% param.noise =  noiseStd;   % stanard deivation of noise 
param.rwSpeed = 1;         % steps each update, default 1


BatchSize = 1;      % minibatch used to to learn
learnType = 'snsm';  % snsm, batch, online, randwalk

noiseVar = 'same';    % same or different noise level for Wij and Mij
param.sigWmax = noiseStd;   % maximum noise std of W
param.sigMmax = noiseStd;   % maximum noise std of M

if strcmp(noiseVar, 'various')
%     noiseVecW = rand(param.Np,1);
    noiseVecW = 10.^(rand(param.Np,1)*2-2);
    param.noiseW = noiseVecW*ones(1,param.Ng)*param.sigWmax;   % noise amplitude is the same for each posterior 
%     noiseVecM = rand(param.Np,1);
    noiseVecM = 10.^(rand(param.Np,1)*2-2);
    param.noiseM = noiseVecM*ones(1,param.Np)*param.sigMmax; 
else
    param.noiseW =  param.sigWmax*ones(param.Np,param.Ng);    % stanard deivation of noise, same for all
    param.noiseM =  param.sigMmax*ones(param.Np,param.Np);   
end



% only used when using "bathc"
numIN  = 10;              % number of inhibitory neurons
Z0 = zeros(numIN,BatchSize);  % initialize interneurons
Y0 = zeros(param.Np, BatchSize); 
V = rand(numIN,param.Np);          % feedback from cortical neurons

gridQuality = 'regular';  % regular or weak


makeAnimation = 0;    % whether make a animation or not

%% generate grid fields
if strcmp(gridQuality,'regular')
    % sample parameters for grid cells
    param.lbds = param.baseLbd*(param.sf.^(0:param.Nlbd-1));   % all the spacings of different modules
    param.thetas =(0:param.Nthe-1)*pi/3/param.Nthe;             % random sample rotations
    param.x0  = (0:param.Nx-1)'/param.Nx*param.lbds;            % offset of x
    param.y0  = (0:param.Ny-1)'/param.Ny*param.lbds;            % offset of

    % generate a Gramian of the grid fields
    gridFields = nan(param.ps^2,param.Ng);
    count = 1;    % concantenate the grid cells
    for i = 1:param.Nlbd
        for j = 1: param.Nthe
            for k = 1:param.Nx
                for l = 1:param.Ny
                    r = [i/param.ps;j/param.ps];
                    r0 = [param.x0(k,i);param.y0(l,i)];
                    gridFields(:,count) = PlaceCellhelper.gridModule(param.lbds(i),...
                        param.thetas(j),r0,param.ps);
                    count = count +1;
                end
            end
        end
    end


    figure
    ha = tight_subplot(3,5);
    sel_inx = randperm(param.Ng, 15);
    for i = 1:15
        gd = gridFields(:,sel_inx(i));
        imagesc(ha(i),reshape(gd,param.ps,param.ps))
        ha(i).XAxis.Visible = 'off';
        ha(i).YAxis.Visible = 'off';
    end

elseif strcmp(gridQuality,'weak')
    % generate weakly-tuned MEC cells
    sig = 2;
    gridFields = PlaceCellhelper.weakMEC(param.ps, param.Ng, sig);
    
    % randomly select 15 of them to show
    figure
    ha = tight_subplot(3,5);
    sel_inx = randperm(param.Ng, 15);
    for i = 1:15
        gd = gridFields(:,sel_inx(i));
        imagesc(ha(i),reshape(gd,param.ps,param.ps))
        ha(i).XAxis.Visible = 'off';
        ha(i).YAxis.Visible = 'off';
    end

end
%% using non-negative similarity matching to learng place fields
% generate input from grid filds

tot_iter = 2e3;   % total interation, default 2e3
sep = 20;

% allW = nan(tot_iter/sep,size(gridFields,2));
% allbias = nan(tot_iter/sep,1);
% all the position input by the grid code
posiGram = eye(param.ps*param.ps);
gdInput = gridFields'*posiGram;  % gramin of input matrix
% gdInput = gridFields'*posiGram*3/sqrt(param.Ng);  % gramin of input matrix


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
    ystart = zeros(param.Np,BatchSize);  % inital of the output
    
    % generate position code by the grid cells    
    for i = 1:tot_iter
        positions = gdInput(:,randperm(param.ps*param.ps,BatchSize));
        states= PlaceCellhelper.nsmDynBatch(positions,ystart, param);
        y = states.Y;
        
        % update weight matrix
%         param.W = (1-param.learnRate)*param.W + param.learnRate*y*positions'/BatchSize;
%         param.M = (1-param.learnRate)*param.M + param.learnRate*y*y'/BatchSize;
        param.W = (1-param.learnRate)*param.W + param.learnRate*y*positions'/BatchSize + ...
            sqrt(param.learnRate)*param.noiseW.*randn(param.Np,param.Ng);
        param.M = max(0,(1-param.learnRate)*param.M + param.learnRate*(y*y')/BatchSize + ...
            sqrt(param.learnRate)*param.noiseM.*randn(param.Np,param.Np));
%         param.M = (1-param.learnRate)*param.M + param.learnRate*y*y'/BatchSize + ...
%             sqrt(param.learnRate)*param.noise*randn(param.Np,param.Np);
%         param.W = max((1-param.learnRate)*param.W + param.learnRate*y*gdInput'/BatchSize,0);
        param.b = (1-param.learnRate)*param.b + param.learnRate*sqrt(param.alpha)*mean(y,2);
%         V = states.V;   % current 

        % store the matrix and weights
%         if mod(i,sep) ==0
%             allW(round(i/sep),:) = param.W;
%             allbias(round(i/sep)) = param.b;
%         end
    end
elseif strcmp(learnType, 'randwalk')
    % position information is delivered as a random walk
    ystart = zeros(param.Np,1);  % inital of the output
    
    % generate position code by the grid cells
    ix = 16;
    iy = 16;
    posiInfo = nan(tot_iter,2);
    for i = 1:tot_iter
        [ix,iy] = PlaceCellhelper.nextPosi(ix,iy,param);
        posiInfo(i,:) = [ix,iy];
        positions = gdInput(:,(iy - 1)*param.ps + ix);  % column-wise storation
%         positions = gdInput(:,randperm(param.ps*param.ps,1));
        
        states= PlaceCellhelper.nsmDynBatch(positions,ystart, param);
        y = states.Y;
        
        % update weight matrix
        param.M = (1-param.learnRate)*param.M + param.learnRate*y*y';
        param.W = (1-param.learnRate)*param.W + param.learnRate*y*positions' + ...
            sqrt(param.learnRate)*param.noise*randn(param.Np,param.Ng);
        param.b = (1-param.learnRate)*param.b + param.learnRate*sqrt(param.alpha)*mean(y,2);

    end
    
end


% estimate the place field of place cells
numProb = 1024;    % number of positions used to estimate the RF

ys_prob = zeros(param.Np,1);
y = zeros(param.Np,numProb);

% make sure the neural dynamcis converge
for i = 1:numProb
    states= PlaceCellhelper.nsmDynBatch(gdInput(:,i),ys_prob, param);
    y(:,i) = states.Y;
end
%% Analysis and plot
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

% for figure plot, select two
figure
hf = tight_subplot(1,4);
for i = 1:4
    pf = y(randperm(param.Np,1),:);
    imagesc(hf(i),reshape(pf,param.ps,param.ps));
    hf(i).XAxis.Visible = 'off';
    hf(i).YAxis.Visible = 'off';
end

% suface plot of an exmaple place field
PMat = reshape(y(randperm(param.Np,1),:),param.ps,param.ps);
figure
surf(PMat)
zlim([0,5])
set(gca,'XTick',[],'YTick',[])

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
    imagesc(hinput(i),reshape(gdInput(:,sel_inx(i)),param.Nlbd*param.Nthe,param.Nx*param.Ny))
    hinput(i).XAxis.Visible = 'off';
    hinput(i).YAxis.Visible = 'off';
end

% input similarity
figure
imagesc(gdInput'*gdInput,[60,110])
colorbar
xlabel('position index','FontSize',24)
ylabel('position index','FontSize',24)
set(gca,'LineWidth',1.5, 'FontSize',20)
title('Input similarity matrix','FontSize',24)

figure
imagesc(gdInput(:,1:50))
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
plSel = randperm(param.Np,min(15,param.Np));
pwhdl = tight_subplot(3,5);
for i = 1:length(plSel)
    imagesc(pwhdl(i),reshape(param.W(plSel(i),:),param.Nlbd*param.Nthe,param.Nx*param.Ny))
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

tot_iter = 2e4;
num_sel = 200;
step = 10;
time_points = round(tot_iter/step);

allW = nan(tot_iter/sep,size(gridFields,2));
allbias = nan(tot_iter/sep,1);

pks = nan(param.Np,time_points);    % store all the peak positions
pkAmp = nan(param.Np,time_points);  % store the peak amplitude
placeFlag = nan(param.Np,time_points); % determine whether a place field

pkCenterMass = nan(param.Np,2,time_points);  % store the center of mass
pkMas = nan(param.Np,time_points);           % store the average ampltude 
% store the weight matrices to see if they are stable
% timeGap = 50;
% allW = nan(param.Np,param.Ng,round(tot_iter/timeGap));
% allM = nan(param.Np,param.Np,round(tot_iter/timeGap));
% allY = nan(param.Np,param.ps^2,round(tot_iter/timeGap)); % store all the population vectors

ampThd = 0.1;   % amplitude threshold, depending on the parameters

% testing data, only used when check the representations
% Xsel = gdInput(:,1:10:end);
Y0 = zeros(param.Np,size(gdInput,2));
Z0 = zeros(numIN,size(gdInput,2));
Yt = nan(param.Np,param.ps*param.ps,time_points);
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
            Yt(:,:,round(i/step)) = states_fixed.Y;

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
            sqrt(param.learnRate)*param.noiseW.*randn(param.Np,param.Ng);  % 9/16/2020
%         param.M = (1-param.learnRate)*param.M + param.learnRate*y*y'/BatchSize;
%         param.W = (1-param.learnRate)*param.W + param.learnRate*y*positions'/BatchSize;
        param.M = max(0,(1-param.learnRate)*param.M + param.learnRate*y*y'/BatchSize + ...
            sqrt(param.learnRate)*param.noiseM.*randn(param.Np,param.Np));
%         param.M = (1-param.learnRate)*param.M + param.learnRate*y*y'/BatchSize + ...
%             sqrt(param.learnRate)*param.noise*randn(param.Np,param.Np);
        param.b = (1-param.learnRate)*param.b + param.learnRate*sqrt(param.alpha)*mean(y,2);
%         param.b = (1-param.learnRate)*param.b + param.learnRate*sqrt(param.alpha)*mean(y,2) + ...
%             sqrt(param.learnRate)*param.noise*randn(param.Np,1);
        
        % store and check representations
        Y0 = zeros(param.Np,size(gdInput,2));
        if mod(i, step) == 0
            states_fixed = PlaceCellhelper.nsmDynBatch(gdInput,Y0, param);
            flag = sum(states_fixed.Y > ampThd,2) > 4;  % only those neurons that have multiple non-zeros
            [~,pkInx] = sort(states_fixed.Y,2, 'descend');
            temp = states_fixed.Y((pkInx(:,1)-1)*param.Np + (1:param.Np)');
            pkAmp(flag,round(i/step)) = temp(flag);
%             pcInx = pkAmp(:,round(i/step)) > ampThd & flag;  % find the place cells
            pks(flag,round(i/step)) = pkInx(flag,1);
            Yt(:,:,round(i/step)) = states_fixed.Y;
            
            % store the center of mass
            [pkCM, aveMass] = PlaceCellhelper.centerMassPks(states_fixed.Y,param, ampThd);
            pkCenterMass(:,:,round(i/step)) = pkCM;
            pkMas(:,round(i/step)) = aveMass;
            
            
%             allW(round(i/sep),:) = param.W(1,:);
%             allbias(round(i/sep)) = param.b(1);
        end
        
%         if mod(i,timeGap) ==0
%             allW(:,:,round(i/timeGap)) = param.W;
%             allM(:,:,round(i/timeGap)) = param.M;
%             allY(:,:,round(i/timeGap)) = states_fixed.Y;
%         end
               
    end

elseif strcmp(learnType, 'randwalk')
    
    ystart = zeros(param.Np,BatchSize);  % inital of the output
    
    % generate position code by the grid cells    
    for i = 1:tot_iter
        [ix,iy] = PlaceCellhelper.nextPosi(ix,iy,param);
%         posiInfo(i,:) = [ix,iy];
        positions = gdInput(:,(iy - 1)*param.ps + ix);
%         positions = gdInput(:,randperm(param.ps*param.ps,BatchSize));
        
        states = PlaceCellhelper.nsmDynBatch(positions,ystart, param);
        y = states.Y;
        
        % noisy update weight matrix
        param.W = (1-param.learnRate)*param.W + param.learnRate*y*positions'/BatchSize + ...
            sqrt(param.learnRate)*param.noise*randn(param.Np,param.Ng);  % 9/16/2020
        param.M = (1-param.learnRate)*param.M + param.learnRate*y*y'/BatchSize;
        param.b = (1-param.learnRate)*param.b + param.learnRate*sqrt(param.alpha)*mean(y,2);
        
        % store and check representations
        % store and check representations
        Y0 = zeros(param.Np,size(gdInput,2));
        if mod(i, step) == 0
            states_fixed = PlaceCellhelper.nsmDynBatch(gdInput,Y0, param);
            flag = sum(states_fixed.Y > ampThd,2) > 4;  % only those neurons that have multiple non-zeros
            [~,pkInx] = sort(states_fixed.Y,2, 'descend');
            temp = states_fixed.Y((pkInx(:,1)-1)*param.Np + (1:param.Np)');
            pkAmp(flag,round(i/step)) = temp(flag);
%             pcInx = pkAmp(:,round(i/step)) > ampThd & flag;  % find the place cells
            pks(flag,round(i/step)) = pkInx(flag,1);
            Yt(:,:,round(i/step)) = states_fixed.Y;
                       
             % store the center of mass
            [pkCM, mass] = PlaceCellhelper.centerMassPks(states_fixed.Y,param, ampThd);
            pkCenterMass(:,:,round(i/step)) = pkCM;
            pkMas(:,round(i/step)) = mass;
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

% make an animation of peak positions
% randomly select 10 neurons
selNeur = randperm(param.Np,min(10,param.Np));
set1 = brewermap(11,'Set1');
frameSep = 5; % only capture every 5 steps, each step  = 2 iterations

epPosiXs = floor(pks(selNeur,:)/param.ps);
epPosiYs = mod(pks(selNeur,:),param.ps); 


if makeAnimation
    figure
    set(gcf,'color','w');
    hold on
    for i = 1:length(selNeur)
        plot(epPosiXs(i,1),epPosiYs(i,1),'o','MarkerSize',20,'MarkerEdgeColor',...
            set1(i,:),'LineWidth',3)
        xlim([0,32])
        ylim([0,32])
        text(epPosiXs(i,1)-0.3,epPosiYs(i,1),num2str(i),'FontSize',20)
    end
    hold off
    box on
    xlabel('x','FontSize',20)
    ylabel('y','FontSize',20)
    set(gca,'LineWidth',1.5,'FontSize',20)

    detlaT = 0.05;
    f = getframe(gcf);
    [im,map] = rgb2ind(f.cdata,256,'nodither');
    k = 1;
    for i = 1:round(size(epPosiXs,2)/frameSep)
        for j=1:length(selNeur)
            plot(epPosiXs(j,i*frameSep),epPosiYs(j,i*frameSep),'o','MarkerSize',20,'MarkerEdgeColor',...
            set1(j,:),'LineWidth',3)
            text(epPosiXs(j,i*frameSep)-0.3,epPosiYs(j,i*frameSep),num2str(j),'FontSize',16)
            hold on
        end
        hold off
        box on
        xlim([0,32])
        ylim([0,32])
        xlabel('x','FontSize',20)
        ylabel('y','FontSize',20)
        set(gca,'LineWidth',1.5,'FontSize',20)
        title(['time = ', num2str(i*frameSep)])
        f = getframe(gcf);
        im(:,:,1,k) = rgb2ind(f.cdata,map,'nodither');
        k = k + 1;

    end
    imwrite(im,map,'scatterPkPosi.gif','DelayTime',detlaT,'LoopCount',inf)
end

%% Estimate the diffusion constant

% estimate the msd
% msds = nan(min(floor(time_points/2),1000),size(pkMas,1));;
% pkMas = pkAmp;
msds = nan(floor(time_points/2),size(pkMas,1));
for i = 1:size(msds,1)
    
    xcoord = squeeze(pkCenterMass(:,1,:));
    ycoord = squeeze(pkCenterMass(:,2,:));
    if param.Np == 1
        t1 = abs(xcoord(i+1:end)' - xcoord(1:end-i)');
        t2 = abs(ycoord(i+1:end)' - ycoord(1:end-i)');
    else
        t1 = abs(xcoord(:,i+1:end,:) - xcoord(:,1:end-i,:));
        t2 = abs(ycoord(:,i+1:end,:) - ycoord(:,1:end-i,:));
    end
    
    dx = min(t1,param.ps - t1);
    dy = min(t2,param.ps - t2);
    msds(i,:) = nanmean(dx.^2 + dy.^2,2)';
end


% linear regression to get the diffusion constant of each neuron
Ds = PlaceCellhelper.fitLinearDiffusion(msds,step,'linear');

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
set(gca,'FontSize',20,'LineWidth',1.5)
xlabel('time','FontSize',24)
ylabel('neuron','FontSize',24)

% time period that a neuron has active place field
temp = pkAmp > ampThd;
tolActiTime = sum(temp,2)/size(pkAmp,2);
avePks = nan(size(pkAmp));
avePks(temp) = pkAmp(temp);
meanPks = nanmean(avePks,2);

figure
plot(meanPks,tolActiTime,'o','MarkerSize',8,'LineWidth',2)
xlabel('Average peak amplitude','FontSize',24)
ylabel('faction of active time','FontSize',24)
set(gca,'FontSize',20,'LineWidth',1.5)

figure
histogram(tolActiTime,20)
xlabel('active time fraction','FontSize',24)
ylabel('count','FontSize',24)
set(gca,'FontSize',20,'LineWidth',1.5)

%% interval of active, silent, shift of RF
% interval length where place fields disappears
ampThd = 0.1;
pkFlags = pkAmp > ampThd;
silentInter = [];  % store the silent interval
activeInter = [];
randActiInter = [];  % compare with random case
actInxPerm = reshape(pkFlags(randperm(size(pkFlags,1)*size(pkFlags,2))),size(pkFlags));
for i = 1:size(pkAmp,1)
    I =  find(pkFlags(i,:));
    temp = diff(I);
    silentInter = [silentInter,temp(temp>1)-1];
    
    Ia = find(pkFlags(i,:)==0);
    temp = diff(Ia);
    activeInter = [activeInter,temp(temp>1)-1];
    
    % for random permutation
    Ib = find(actInxPerm(i,:)==0);
    temp = diff(Ib);
    randActiInter = [randActiInter,temp(temp>1)-1];
end

% tabulate the results
randActTab = tabulate(randActiInter);
actTab = tabulate(activeInter);

figure
hold on
plot(actTab(:,1),actTab(:,3)/actTab(1,3),'LineWidth',3)
plot(randActTab(:,1),randActTab(:,3)/randActTab(1,3),'LineWidth',3)
hold off
box on
legend('Model','Random')
xlim([1,20])
xlabel('$\Delta t$','Interpreter','latex','FontSize',28)
ylabel('Fraction','FontSize',28)
set(gca,'FontSize',24,'LineWidth',1,'XScale','log')

% histogram of all the silent interval
% also fit an exponential distribution
pd = fitdist(silentInter(:),'Exponential');
xval=0:0.5:50;
yfit = pdf(pd,xval);

silentIt = figure;
hold on
set(silentIt,'color','w','Units','inches')
pos(3)=3.8;  %17.4; %8.5;%pos(3)*1.5;
pos(4)=2.8;%pos(4)*1.5;
set(silentIt,'Position',pos)

histogram(silentInter(:),'Normalization','pdf')
plot(xval,yfit,'LineWidth',2)
hold off
box on
xlim([0,60])
lg = legend('simulation','exponential fit');

xlabel('Silent interval','FontSize',16)
ylabel('Pdf','FontSize',16)
set(gca,'LineWidth',1,'FontSize',16)

% prefix = 'pc2D_Silent_Interval';
% saveas(silentIt,[sFolder,filesep,prefix,'.fig'])
% print('-depsc',[sFolder,filesep,prefix,'.eps'])


% histogram of the active period
activeIt = figure;
set(activeIt,'color','w','Units','inches')
pos(3)=3.8;  %17.4; %8.5;%pos(3)*1.5;
pos(4)=2.8;%pos(4)*1.5;
set(activeIt,'Position',pos)
histogram(activeInter(:))
xlim([0,100])
xlabel('Active interval','FontSize',16)
ylabel('Count','FontSize',16)
set(gca,'LineWidth',1,'FontSize',16)


% correlation between RF shift and total time that are silent

% Diffusion constant vs average noise level
tot_noise_var = param.noiseW(:,1).^2 + param.noiseM(:,1).^2;
eff_acti_inx = ~isnan(Ds);

% fit a linear regression model
mdl = fitlm(tot_noise_var(eff_acti_inx),Ds(eff_acti_inx));
fit_stat = anova(mdl,'summary');
fh = plot(mdl);
ah = fh.Parent;
fh(1).Marker = 'o';
fh(1).Color = 'k';
title('')
xlabel('$\sigma_W^2 + \sigma_M^2$','Interpreter','latex','FontSize',24)
ylabel('$D$','Interpreter','latex','FontSize',24)
set(gca,'YScale','linear','XScale','linear','FontSize',24)



figure
plot(tot_noise_var(eff_acti_inx),Ds(eff_acti_inx),'o','MarkerSize',10)
xlabel('$\sigma_W^2 + \sigma_M^2$','Interpreter','latex','FontSize',24)
ylabel('$D$','Interpreter','latex','FontSize',24)
set(gca,'YScale','linear','XScale','linear','FontSize',24)


% loglog plot of diffusion constant vs synaptic noise
figure
loglog(tot_noise_var(eff_acti_inx),Ds(eff_acti_inx),'o','MarkerSize',8)
xlabel('$\sigma_W^2 + \sigma_M^2$','Interpreter','latex','FontSize',24)
ylabel('$D$','Interpreter','latex','FontSize',24)
set(gca,'FontSize',24)

%% SAVE THE DATA OR NOT
save_folder = fullfile(pwd,'data', filesep,'revision');
save_data_name = fullfile(save_folder, ['pc_2D_various_noise_',date,'.mat']);
save(save_data_name,'-v7.3')

%% Publication ready figures
% this part polish some of the figures and make them publication ready
% define all the colors
sFolder = './figures';
figPre = 'placeCell2D_0412_2022_variousSig';   % this should change according to the task

nc = 256;   % number of colors
spectralMap = brewermap(nc,'Spectral');
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


% ************************************************************
% Diffusion constant vs the synaptic noise level
% ************************************************************
% plot when we use various noise level for different neurons
if strcmp(noiseVar, 'various')
   
    % first loglog plot
    f_D_sig = figure;
    pos(3)=3.2; pos(4)=2.8;
    set(f_D_sig,'color','w','Units','inches','Position',pos)
    
    loglog(tot_noise_var(eff_acti_inx),Ds(eff_acti_inx),'o','MarkerSize',6,...
        'MarkerFaceColor',greys(1,:),'MarkerEdgeColor',greys(9,:),'LineWidth',1)
  
    
    xlabel('$\sigma_W^2 + \sigma_M^2$','Interpreter','latex','FontSize',16)
    ylabel('$D$','Interpreter','latex','FontSize',16)
    set(gca,'XTick',10.^(-7:-4))

    prefix = [figPre, 'D_sigmas'];
    saveas(f_D_sig,[sFolder,filesep,prefix,'.fig'])
    print('-depsc',[sFolder,filesep,prefix,'.eps'])
  
    % make a linear regression and plot the confidence interval and R^2
    sig_range = 10.^(-7:0.1:-3.8)';
    D_pred =  predict(mdl,sig_range);
    
    f_D_sig_linear = figure;
    set(f_D_sig_linear,'color','w','Units','inches','Position',pos)
    
    plot(tot_noise_var(eff_acti_inx),Ds(eff_acti_inx),'o','MarkerSize',6,...
       'MarkerEdgeColor',greys(9,:))
    hold on
    plot(sig_range,D_pred,'LineWidth',1.5,'Color',blues(9,:))
    hold off
    xlabel('$\sigma_W^2 + \sigma_M^2$','Interpreter','latex','FontSize',16)
    ylabel('$D$','Interpreter','latex','FontSize',16)
    
    R2 = num2str(round(mdl.Rsquared.Ordinary*100)/100);
    pvalue = num2str(fit_stat.pValue(2));
    annotation('textbox', [.1 .8,.85  .1],...
    'String', ['R^2 =',R2,'; p-value =',pvalue],'BackgroundColor','none','Color','k',...
    'LineStyle','none','fontsize',10,'HorizontalAlignment','Center');

    prefix = [figPre, 'D_sigmas_linearRegr'];
    saveas(f_D_sig_linear,[sFolder,filesep,prefix,'.fig'])
    print('-depsc',[sFolder,filesep,prefix,'.eps'])
    
end


% ************************************************************
% Diffusion constant vs active time
% ************************************************************
DiffuAmpFig = figure;
pos(3)=3.5;  pos(4)=2.8;
set(DiffuAmpFig,'color','w','Units','inches','Position',pos)
plot(tolActiTime(eff_acti_inx),Ds(eff_acti_inx),'o','MarkerSize',4,...
    'MarkerEdgeColor',greys(9,:),'LineWidth',1)
xlabel('Fraction of active time','FontSize',16)
ylabel('$D$','Interpreter','latex','FontSize',16)
set(gca,'FontSize',16,'LineWidth',1)

prefix = [figPre, 'D_active_time'];
saveas(DiffuAmpFig,[sFolder,filesep,prefix,'.fig'])
print('-depsc',[sFolder,filesep,prefix,'.eps'])

% ylim([0,2.5])



% ************************************************************
% peak position, using time point 1
% ************************************************************
temp = [ceil(pks(:,1)/param.ps),mod(pks(:,1),param.ps)]/param.ps;
pkVec = temp(~isnan(temp(:,1)),:);

% define figure size
f_pkPosi = figure;
set(f_pkPosi,'color','w','Units','inches')
pos(3)=3.2;  %17.4; %8.5;%pos(3)*1.5;
pos(4)=2.8;%pos(4)*1.5;
set(f_pkPosi,'Position',pos)
% histogram(activeInter(:))
% xlim([0,100])
plot(pkVec(:,1),pkVec(:,2),'o','MarkerSize',4,'MarkerFaceColor',greys(9,:),...
    'MarkerEdgeColor',greys(9,:))
xlabel('X position','FontSize',16)
ylabel('Y position','FontSize',16)
set(gca,'LineWidth',1,'FontSize',16)

prefix = [figPre, 'pkPosi'];
saveas(f_pkPosi,[sFolder,filesep,prefix,'.fig'])
print('-depsc',[sFolder,filesep,prefix,'.eps'])



% ************************************************************
% peak amplitude of example neuron
% ************************************************************
inx = randperm(param.Np,1);  % select one neuron to plot, 10 for old data
inx = 155;  %155
temp = pkAmp(inx,:);
temp(isnan(temp)) = 0;
times = (1:length(temp))*step;   % iterations

f_pkAmpTraj = figure;
set(f_pkAmpTraj,'color','w','Units','inches')
pos(3)=3.2;  %17.4; %8.5;%pos(3)*1.5;
pos(4)=2.8;%pos(4)*1.5;
set(f_pkAmpTraj,'Position',pos)

plot(times',temp','LineWidth',1.5,'Color',blues(9,:))
xlabel('Time','FontSize',16)
ylabel('Peak Amplitude','FontSize',16)
set(gca,'LineWidth',1,'FontSize',16)

prefix = [figPre, 'pkAmp'];
saveas(f_pkAmpTraj,[sFolder,filesep,prefix,'.fig'])
print('-depsc',[sFolder,filesep,prefix,'.eps'])

% ************************************************************
% peak amplitude of all neurons across all time
% ************************************************************
f_pkAmpall = figure;
set(f_pkAmpall,'color','w','Units','inches')
pos(3)=3.2;  %17.4; %8.5;%pos(3)*1.5;
pos(4)=2.8;%pos(4)*1.5;
set(f_pkAmpall,'Position',pos)

pkAllData = pkAmp(~isnan(pkAmp));
histogram(pkAllData,'Normalization','pdf')
xlim([0,8])
xlabel('peak amplitude','FontSize',16)
ylabel('pdf','FontSize',16)
set(gca,'LineWidth',1,'FontSize',16)


% ************************************************************
% Fraction of active neurons
% ************************************************************
f_acti = figure;
set(f_acti,'color','w','Units','inches')
pos(3)=3.3;  
pos(4)=2.8;
set(f_acti,'Position',pos)
actiFrac = mean(pks > 0,1);

plot(times',actiFrac,'LineWidth',1.5)
xlabel('Time','FontSize',16)
ylabel('Active fraction','FontSize',16)
ylim([0,1])
set(gca,'LineWidth',1,'FontSize',16)

prefix = [figPre, 'fracActive'];
saveas(f_acti,[sFolder,filesep,prefix,'.fig'])
print('-depsc',[sFolder,filesep,prefix,'.eps'])


% ************************************************************
% distribution of active time
% ************************************************************
f_actiTime = figure;
set(f_actiTime,'color','w','Units','inches')
pos(3)=3.5;  
pos(4)=2.8;
set(f_actiTime,'Position',pos)
actiTime = mean(pks > 0,2);

histogram(actiTime,20);
xlabel('Active time','FontSize',16)
ylabel('Count','FontSize',16)
set(gca,'LineWidth',1,'FontSize',16)


% ************************************************************
% peak position of example neuron
% ************************************************************
% epInx = randperm(param.Np,1);  % randomly slect
epInx = 105;
epPosi = [floor(pks(epInx,:)/param.ps);mod(pks(epInx,:),param.ps)]+randn(2,size(pks,2))*0.1;

specColors = brewermap(size(epPosi,2), 'Spectral');

% colors indicate time
f_pkPosi = figure;
set(f_pkPosi,'color','w','Units','inches')
pos(3)=3.2;  
pos(4)=2.8;
set(f_pkPosi,'Position',pos)
hold on

for i=1:size(pks,2)
    plot(epPosi(1,i),epPosi(2,i),'o','MarkerSize',4,'MarkerEdgeColor',...
        specColors(i,:),'LineWidth',1.5)
end
hold off
box on
xlabel('x position','FontSize',16)
ylabel('y position','FontSize',16)
set(gca,'FontSize',16,'LineWidth',1)

% ************************************************************
% average peak amplitude vs time of active
% ************************************************************

temp = pkAmp > ampThd;
tolActiTime = sum(temp,2)/size(pkAmp,2);
avePks = nan(size(pkAmp));
avePks(temp) = pkAmp(temp);
meanPks = nanmean(avePks,2);

f_ampActiTime= figure;
set(f_ampActiTime,'color','w','Units','inches')
pos(3)=3.5;  
pos(4)=2.8;
set(f_ampActiTime,'Position',pos)
plot(meanPks,tolActiTime,'o','MarkerSize',4,...
    'MarkerEdgeColor',greys(9,:),'LineWidth',1.5)
xlabel('Average peak amplitude','FontSize',16)
ylabel('Faction of active time','FontSize',16)
set(gca,'FontSize',16,'LineWidth',1)

prefix = [figPre, 'pkAmp_actiTime'];
saveas(f_ampActiTime,[sFolder,filesep,prefix,'.fig'])
print('-depsc',[sFolder,filesep,prefix,'.eps'])

% ************************************************************
% distribution of peak shift
% ************************************************************
% prepare the data
shiftStat = nan(param.Np,2);  %store the mean and std of peak shift
allShiftDist = [];
for i = 1:param.Np
    flag = find(pks(i,:) > 0);
    temp = pks(i,flag);
%     tsq = zeros(1,size(pks,2));
%     tsq(flag) = flag;
%     jumpFlag = find(diff(tsq) > 1) + 1;
    jumpFlag = find(diff(flag)>1);
    
    
    xs = floor(temp/param.ps);
    ys = mod(temp,param.ps);
    shiftFlag = find(diff(temp) ~= 0); % whether shift or not
    
    jointFlag = union(shiftFlag,jumpFlag);
    
    dxs = xs(jointFlag + 1) - xs(jointFlag);
    dys = ys(jointFlag + 1) - ys(jointFlag);
    shiftEcul = sqrt(dxs.^2 + dys.^2)/param.ps;
    shiftStat(i,:) = [mean(shiftEcul),std(shiftEcul)]; 
    allShiftDist = [allShiftDist,shiftEcul];
end

% active peak amplitude and peak shift
f_pkShift = figure;
set(f_pkShift,'color','w','Units','inches')
pos(3)=3.5;  
pos(4)=2.8;
set(f_pkShift,'Position',pos)
plot(meanPks,shiftStat(:,1),'o','MarkerSize',4,...
    'MarkerEdgeColor',greys(9,:),'LineWidth',1.5)
xlabel('Average peak amplitude','FontSize',16)
ylabel('$\langle \Delta r \rangle$','Interpreter','latex','FontSize',16)
set(gca,'FontSize',16,'LineWidth',1)
ylim([0,0.4])

prefix = [figPre, 'pkShit_actiTime'];
saveas(f_pkShift,[sFolder,filesep,prefix,'.fig'])
print('-depsc',[sFolder,filesep,prefix,'.eps'])



figure
ecdf(allShiftDist)
histogram(allShiftDist(allShiftDist>0.1))

% ************************************************************
% Change of population vectors, measured by correlation
% ************************************************************

% fmSel = randperm(size(Yt,2),1); % randomly slect three frames to consider
pvCorr = zeros(size(Yt,3),size(Yt,2)); 
% [~,neuroInx] = sort(peakInx(:,inxSel(1)));

for i = 1:size(Yt,3)
    for j = 1:size(Yt,2)
        temp = Yt(:,j,i);
        C = corrcoef(temp,Yt(:,j,1));
        pvCorr(i,j) = C(1,2);
    end
end

f_pvCorr = figure;
set(f_pvCorr,'color','w','Units','inches')
pos(3)=3.5;  
pos(4)=2.8;
set(f_pvCorr,'Position',pos)
fh = shadedErrorBar((1:size(pvCorr,1))',pvCorr',{@mean,@std});
box on
set(fh.edge,'Visible','off')
fh.mainLine.LineWidth = 3;
fh.mainLine.Color = blues(10,:);
fh.patch.FaceColor = blues(7,:);
% ylim([0.25,1])
xlim([0,1e3])
xlabel('Time','FontSize',16)
ylabel('PV correlation','FontSize',16)
set(gca,'FontSize',16,'LineWidth',1)

prefix = [figPre, 'pvCorrCoef'];
saveas(f_pvCorr,[sFolder,filesep,prefix,'.fig'])
print('-depsc',[sFolder,filesep,prefix,'.eps'])




% ************************************************************
% Representational similarity across time
% ************************************************************
% for better viusalization, only use part of the data
% gray color maps
selInx = 1:10:1024;
Y1= Yt(:,:,1);
Y2 = Yt(:,:,400);
SM1 = Y1'*Y1;
SM2 = Y2'*Y2;
% SM1 = Y1(:,selInx)'*Y1(:,selInx);
% SM2 = Y2(:,selInx)'*Y2(:,selInx);

f_sm1 = figure;
set(f_sm1,'color','w','Units','inches')
pos(3)=3.5;  
pos(4)=2.5;
set(f_sm1,'Position',pos)

imagesc(SM1,[0,17]);
% colormap(GreyMap)
cb = colorbar;
set(cb,'FontSize',12)
set(gca,'XTick','','YTick','','LineWidth',0.5)

prefix = [figPre, 'sm1'];
saveas(f_sm1,[sFolder,filesep,prefix,'.fig'])
print('-depsc',[sFolder,filesep,prefix,'.eps'])

f_sm2 = figure;
set(f_sm2,'color','w','Units','inches')
pos(3)=3.5;  
pos(4)=2.5;
set(f_sm2,'Position',pos)

% imagesc(SM2,[0,25]);
imagesc(SM2,[0,17]);
% colormap(GreyMap)
cb = colorbar;
set(cb,'FontSize',12)
set(gca,'XTick','','YTick','','LineWidth',0.5)
prefix = [figPre, 'sm2'];
saveas(f_sm2,[sFolder,filesep,prefix,'.fig'])
print('-depsc',[sFolder,filesep,prefix,'.eps'])


%% Overlap of place field
tp = size(Yt,3);  % totoal time point
aveOverLap = nan(tp,1);
aveRFsize = nan(tp,1);
aveNearestDist = nan(size(pkCenterMass,3),1);
for i = 1:size(Yt,3)
    Y = Yt(:,:,i);  % response at time point i
    corrY = Y*Y';
    corrY_bar = corrY - diag(diag(corrY));
    aveOverLap(i) = mean(corrY_bar(:));
    aveRFsize(i) = mean(sum(Y.^2,2));
    
    % distance of nearest neighbor RF
    actCenterMass = pkCenterMass(pkFlags(:,i),:,i);
    pd = pdist(actCenterMass);
    euclDistM = squareform(pd);   % distance matrix
    sortedDist = sort(euclDistM,2,'ascend');
    aveNearestDist(i) = mean(sortedDist(:,2)); % the frist column are zeros
end

aveNearestDist = nan(size(pkCenterMass,3),1);
for i = 1:size(pkCenterMass,3)
    cmPosi = pkCenterMass(:,:,i);
    pd = pdist(cmPosi(~isnan(cmPosi(:,1)),:));
    euclDistM = squareform(pd);   % distance matrix
    sortedDist = sort(euclDistM,2,'ascend');
    aveNearestDist(i) = mean(sortedDist(:,2)); % the frist column are zeros  
end

%% Centroid trajectory

sepSel = 5;
numPoints = round(size(pkCenterMass,3)/sepSel);
bluesSpec = flip(brewermap(numPoints,'blues'));
PuRdSpec = flip(brewermap(numPoints,'PuRd'));
sepctrumColors = {bluesSpec,PuRdSpec};

neurSel = randperm(param.Np,2);

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

%%
figure
imagesc(pkAmp'*pkAmp)

% fraction of cells that are place cells
figure
plot((1:size(pkAmp,2))*step,mean(pkAmp > ampThd,1),'LineWidth',2)
xlabel('$t$','Interpreter','latex','FontSize',24)
ylabel('fraction of place cell','FontSize',24)
set(gca,'FontSize',20,'LineWidth',1.5,'YLim',[0,1])

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
    for j = 1:size(pks,1)
        if ~isnan(pks(j,tps(i)))
            pkMat(pks(j,tps(i))) = 1;
        end
    end
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
z = shiftEcul(~isnan(shiftEcul));
figure
histogram(z,30)
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
% chgW = nan(size(allW,3)-1,1);
% chgM = nan(size(allM,3)-1,1);
% for i = 1:size(allW,3)-1
%     chgW(i) = norm(allW(:,:,i+1) - allW(:,:,1))/norm(allW(:,:,1));
%     chgM(i) = norm(allM(:,:,i+1) - allM(:,:,1))/norm(allM(:,:,1));
% end


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
