% this porgram test the analytical calculation of single step centroid
% shift
% we start by assume a well defined truncated cosine RF, as well as a
% smooth lateral connection matrix M
% We then random select one input and do a stochastic update, either with
% input noise or with synaptic noise
% We aim to test if the correlation of centroid shift due to this update
% has a distance-dependence as we expected

clear
close all

%% graphics settings
sFolder = './figures';

blues = brewermap(11,'Blues');
reds = brewermap(11,'Reds');
greys = brewermap(11,'Greys');

%% Numerically solve the optimal solution of NSM0
% this is the method from Manifold tiling paper
N = 628;
ps = 628;
lbd = 0.1;
alp = 0;

radius = 1;
sep = 2*pi/ps;
ags =sep:sep:2*pi;
% X = [radius*cos(sep:sep:2*pi);radius*sin(sep:sep:2*pi)];
X = [radius*cos(2*pi*(1:ps)/ps);radius*sin(2*pi*(1:ps)/ps)];

% initialize RF
psi = pi/3;  % initial guess of the RF
mu = 0.1;   % initial guess of the amplitude
centers = (1:N)*round(ps/N)*2*pi/ps;  % position of centroid
RF = @(x,phi) mu*max(cos(x-phi) - cos(psi),0);

allRFs = nan(ps,N);  % store all the RFs
for i = 1:N
    allRFs(:,i) = RF(ags(1:end)',centers(i));
end
Y0 = allRFs';

% interate until converge
MaxIter = 1e4;
ErrTol = 1e-10;
dt = 0.001;
err = inf;
count = 1;
D = X'*X;
Q = Y0'*Y0;
% Q = zeros(ps,ps);
Yold = Y0;

sep = round(ps/N);
Y = Y0;
y0 = allRFs(:,50);
yold = y0;
uyold = zeros(ps,1);

% predefine the centroid position
RFctr0 = (1:N)*sep/ps*2*pi;   % in unit of radius

while count < MaxIter && err > ErrTol
%     uy = dt*(-yold*lbd + max((D - alp*eye(ps) - Q)*yold,0));
%     y = yold + uy;
    uy = dt*((D - alp*eye(ps) - Q)*yold - lbd*yold);
%     uy = dt*((D - alp*eye(ps))*yold - lbd*yold);
%     uy = uyold + dt*(-uyold + (D - alp*eye(ps) - Q)*yold/lbd);
%     y = max(uy,0);
%     ymax = max(yold);
    y = max(yold + uy,0);
    
    
%     uY = dt*(-Yold*lbd + Yold*(D - alp*eye(ps) - Yold'*Yold));
%     Y = max(Yold + uY,0);
%     Q = Y*Y';
%     H = D - alp*eye(ps) - Q;
%     [V,S] = eig(H);
%     ys = max(V(:,end),0);
%     [~,inx] = sort(y,'descend');
%     ys_shift = circshift(y,ps - inx(1) + 1);
    
    % set the new Y
    for i = 1:N
        Y(i,:) = circshift(y',sep*(i-1));
    end
    Q = Y'*Y;
    
%     Y = max((D - alp*eye(ps) - Q)*Yold,0)/lbd;
%     Y = MantHelper.quadprogmNSM0(X,Yold,alp,lbd);
%     Q = Y*Y';
%     err = norm(Yold - Y ,'fro')/(norm(Yold,'fro')+ 1e-12);
%     Yold = Y;
    err = norm(yold - y ,'fro')/(norm(yold,'fro')+ 1e-16);
    yold = y;
    uyold = uy;
    count = count + 1;
end



figure
imagesc(Y)

figure
imagesc(Y'*Y)

figure
plot(Y(100*(1:5)-20,:)')
xlabel('Position')
ylabel('Response')

figure
plot(Y(50,:))

% define forward and recurrent matrix
W = Y*X'/ps;
M  = Y*Y'/ps;
Mbar = M - diag(diag(M));

% check if the solution is self-consistent
b = sqrt(alp)*mean(Y,2);
I = W*X - alp*b - Mbar*Y;
Z = max(I,0);
figure
imagesc(Z)

figure
imagesc(Y,[0,0.12])
colorbar
set(gca,'XTick',[1,round(size(Y,2)/2),size(Y,2)],'XTickLabel',{'0','\pi','2\pi'})
xlabel('$\theta$','Interpreter','latex')
ylabel('Neuron')

% normalize the response
lbd2 = max(Z(50,:))/max(Y(50,:)) - M(1,1);
figure
hold on
plot(Y(50,:))
plot(Z(50,:)/(lbd2 + M(1,1)))
hold off

% set the model parameters
param.lbd1 = 0.0;              % 0.15 for 400 place cells and 5 modes
param.lbd2 = lbd2;              % 0.05 for 400 place cells and 5 modes
param.W = W;
param.M = M;
param.b = zeros(N,1);
param.alpha = alp;

param.gy = 0.01;

Xsel = X(:,4:4:end);  % only select partial data
Y0 = Y(:,4:4:end,:);
% Ys = nan(N,size(Xsel,2));
% for j = 1:size(Xsel,2)
%     Ys(:,j) = MantHelper.quadprogamYfixed(Xsel(:,j),param);
% end
I2 = W*Xsel - alp*b - Mbar*Y0;
figure
imagesc(max(I2,0))
% runnig neural dynamics
% y0 = Y(10:10:end,:)';
[states, param] = MantHelper.nsmDynBatch(Xsel,Y0, param);
imagesc(states.Y)

Y0 = states.Y; 
%% One step update
n = 2;   % input dimension
k = N;   % number of neurons 
param.noise = 0;  % adding noise
param.learnRate = 0.1;   % default 0.05
BatchSize = 1;
learnType = 'snsm';  % only happen in this section
angularVel = 1/80;   % angular velocity
tot_iter = 1e3;
Wold = param.W;
Mold = param.M;
bold = param.b;

% testing data, only used when check the representations
Xsel = X(:,4:4:end);        % modified on 2/23/2021 
ampThd = 0.05;       % threhold of place cell, 2/23/2021

centroidRF = nan(N,tot_iter);  % centroid of RF


if strcmp(learnType, 'snsm')
    for i = 1:tot_iter
        y0 = 0.1*rand(N,BatchSize);
        inx = randperm(ps,BatchSize);
%         inx = max(1,mod(i,t));
        x = X(:,inx);  % randomly select one input
%         y = MantHelper.quadprogamYfixed(x,params);
        [states, param] = MantHelper.nsmDynBatch(x,y0, param);
        y = states.Y;
        % store and check representations
        
        % update the weight
        param.W = (1-param.learnRate)*param.W + param.learnRate*y*x'/BatchSize...
            +sqrt(param.learnRate)*param.noise*randn(k,n);        
        param.M = max((1-param.learnRate)*param.M + param.learnRate*y*y'/BatchSize + ...
            sqrt(param.learnRate)*param.noise*randn(k,k),0);
        param.b = (1-param.learnRate)*param.b + param.learnRate*sqrt(param.alpha)*mean(y,2);
        
        % new RFs
%         Ys = nan(k,size(Xsel,2));
%         Y0 = 0.1*rand(k,size(Xsel,2));
        [states, param] = MantHelper.nsmDynBatch(Xsel,Y0, param);
        Ys = states.Y;
%         y = states.Y;
%         for j = 1:size(Xsel,2)
%             Ys(:,j) = MantHelper.quadprogamYfixed(Xsel(:,j),params);
%         end
%         states_fixed = MantHelper.nsmDynBatch(Xsel,Y0,params);
%         flag = sum(states_fixed.Y > ampThd,2) > 5;  % only those neurons that have multiple non-zeros  
        flag = sum(Ys > ampThd,2) > 3;  % only those neurons that have multiple non-zeros
        % centroids of RF
        centroidRF(:,i) = MantHelper.nsmCentroidRing(Ys,flag);
%         centroidRF(:,round(i/step)) = MantHelper.nsmCentroidRing(states_fixed.Y,flag);
        
        % set the weights back
        param.W = Wold;
        param.M = Mold;
        param.b = bold;
            
    end 
elseif strcmp(learnType, 'inputNoise')
    for i = 1:tot_iter
        y0 = 0.1*rand(N,1);
        ix = randi(round(1/angularVel),1);  % random select one input
        x = [cos(angularVel*ix*2*pi);sin(angularVel*ix*2*pi)] + randn(2,BatchSize)*param.noise;  % randomly select one input and noise
        
        y = MantHelper.quadprogamYfixed(x,param);
        
       % one step weight update
        param.W = (1-param.learnRate)*param.W + param.learnRate*y*x'/BatchSize;
        param.M = (1-param.learnRate)*param.M + param.learnRate*y*y'/BatchSize;
        param.b = (1-param.learnRate)*param.b + param.learnRate*sqrt(param.alpha)*mean(y,2);
           
        % using quadratic programing to find the fixed points
        Ys = nan(k,size(Xsel,2));
        for j = 1:size(Xsel,2)
            Ys(:,j) = MantHelper.quadprogamYfixed(Xsel(:,j),param);
        end
        flag = sum(Ys > ampThd,2) > 3;  % only those neurons that have multiple non-zeros

        % centroids of RF
        centroidRF(:,i) = MantHelper.nsmCentroidRing(Ys,flag);
        
        % set the weights back
        param.W = Wold;
        param.M = Mold;
        param.b = bold;
    end
end

% compare original and shfited RFs
% selInx = 100:200:600;
selInx = [240,380,428];
colors = brewermap(5,'Set1');
figure
hold on
for i = 1:length(selInx)
%     plot(Y(5:5:end,selInx(i)),'Color',colors(i,:))
    plot(Y(selInx(i),4:4:end),'Color',colors(i,:))
    plot(Ys(selInx(i),:),'--','Color',colors(i,:))
end
hold off
box on
xlabel('Position')
ylabel('Response')
set(gca,'XTick',[1,round(size(Ys,2)/2),size(Ys,2)],'XTickLabel',{'0','\pi','2\pi'})

% calculate distance dependent correlations
% initial pairwise-distance
actInx = ~isnan(RFctr0);

selCntRF = centroidRF(actInx,1:end)/size(Xsel,2)*2*pi - RFctr0(actInx)';
% ix = ~isnan(selCntRF);
% selCntRF = mod(selCntRF(ix),-sign(selCntRF(ix))*2*pi); 
ix = abs(selCntRF) > pi;
selCntRF(ix) = mod(selCntRF(ix),-sign(selCntRF(ix))*2*pi); 

selCntRF(isnan(selCntRF))=0;
% 
Z = selCntRF*selCntRF'/size(selCntRF,2);
% pd0 = squareform(pdist(RFctr0(actInx)')/ps*2*pi);
pd0 = squareform(pdist(RFctr0(actInx)'));


inx = abs(pd0) > pi;
pd0(inx) = mod(pd0(inx),-sign(pd0(inx))*2*pi); 

figure
plot(abs(pd0(:)),Z(:),'o')
% ylim([-1,1])
xlabel('Distance')
ylabel('$\langle \Delta r_1\Delta r_2\rangle$','Interpreter','latex')
set(gca,'XTick',[1,round(size(Ys,2)/2),size(Ys,2)],'XTickLabel',{'0','\pi','2\pi'})


% distribution of all the single step shift
figure
histogram(selCntRF(selCntRF~=0),'Normalization','pdf')
xlabel('Centroid shift')
ylabel('Pdf')
% xlim([-0.2,0.2])
% set(gca,'XTick',[-pi/20,0,pi/20],'XTickLabel',{'-\pi/20','0','\pi/20'})


% heatmap to show the purturbation
figure
imagesc(Ys,[0,0.1])
colorbar
xlabel('$\theta$','Interpreter','latex')
ylabel('Neuron')
set(gca,'XTick',[1,round(size(Ys,2)/2),size(Ys,2)],'XTickLabel',{'0','\pi','2\pi'})

figure
hold on
plot(sum(Ys,2),'Color','red')
yl = ylim;
% plot([inx;inx],yl','--','Color','red')
hold off
box on
xlabel('Neuron')
ylabel('Mass of RF')
% set(gca,'XTick',[1,round(size(Ys,2)/2),size(Ys,2)],'XTickLabel',{'0','\pi','2\pi'})

figure
hold on
plot(sum(Ys,1))
yl = ylim;
xp0 = round(inx/4);  % sample position
plot([xp0;xp0],yl','--')
hold off
box on
xlabel('Position')
ylabel('Total mass')
set(gca,'XTick',[1,round(size(Ys,2)/2),size(Ys,2)],'XTickLabel',{'0','\pi','2\pi'})


figure
imagesc(Wold)
colorbar
set(gca,'XTick',[1,2])


figure
imagesc(param.W)
colorbar
set(gca,'XTick',[1,2])

figure
imagesc(param.W - Wold)
colorbar
set(gca,'XTick',[1,2])

figure
imagesc(Mold)
colorbar

figure
imagesc(param.M)
colorbar

figure
imagesc(param.M - Mold)
colorbar

figure
plot(y)
ylabel('Response')
xlabel('Position')
set(gca,'XTick',[1,round(length(y)/2),length(y)],'XTickLabel',{'0','\pi','2\pi'})

%% parameters of the model

N = 200;
ps = 1e3;   % number of point on the ring manifold
alp = 0;
% mu = 1/sqrt(1+3*N*(4+pi^2)/16/pi^2);   % assume every neuron has the same aplitude
psi = 1;   % the width
mu = 0.25548;

% generate all the input
radius = 1;
sep = 2*pi/ps;
ags =sep:sep:2*pi;
X = [radius*cos(sep:sep:2*pi);radius*sin(sep:sep:2*pi)];

% pre-defined RFs
centers = (1:N)*round(ps/N)*2*pi/ps;  % position of centroid
RF = @(x,phi) mu*max(cos(x-phi) - cos(psi),0);

maxVar = @(y) mu^2/4/pi*(4*y + 2*y*cos(2*y) - 3*sin(2*y));

allRFs = nan(ps,N);  % store all the RFs
for i = 1:N
    allRFs(:,i) = RF(ags(1:end)',centers(i));
end

% define the forward matrix W
W = allRFs'*X(:,1:end)'/ps;

figure
imagesc(W)

% define the self-consistent lateral matrix M
M = allRFs'*allRFs/ps;
Mbar = M - diag(diag(M));

% test if the above matrices are self-consistent
param.lbd1 = 0.0;              % 0.15 for 400 place cells and 5 modes
param.lbd2 = 0.0;              % 0.05 for 400 place cells and 5 modes
param.W = W;
param.M = M;
param.b = zeros(N,1);
param.alpha = 0;

param.gy = 0.05;


Xsel = X(:,10:10:end);  % only select partial data
% Ys = nan(N,size(Xsel,2));
% for j = 1:size(Xsel,2)
%     Ys(:,j) = MantHelper.quadprogamYfixed(Xsel(:,j),param);
% end

% runnig neural dynamics
y0 = allRFs(10:10:end,:)';
[states, param] = MantHelper.nsmDynBatch(Xsel,y0, param);


figure
imagesc(Mbar*allRFs')

figure
imagesc(M)

figure
plot(M(100,:))

figure
imagesc(W*X(:,2:end))

figure
imagesc((W*X(:,2:end)- Mbar*allRFs')/maxVar(psi))
z = (W*X(:,2:end)- Mbar*allRFs')/maxVar(psi);

figure
plot(max(z(100,:),0))
hold on
plot(allRFs(:,100))
hold off

figure
imagesc(allRFs)
phi = 0.5;
rf = RF(-pi:0.01:pi,phi);
figure
plot(-pi:0.01:pi,rf)