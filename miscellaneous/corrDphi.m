% show that the cross term is negnigible of dphi

[k,N,samples] = size(Yt);
drs = Yt(:,2:end,:) - Yt(:,1:(N-1),:);
lens = sum(Yt(:,2:end,:).^2,1);  % square of vectors
angVel = cross(Yt(:,1:(N-1),:), drs, 1)./lens; % angular velocity

selL = 100;
Num = floor(N/selL)-1;

allDphi = nan(samples,Num);
Diag = nan(samples,Num);
temp = ones(selL);
mask = tril(temp,-1);

for j = 1:Num
    angPart = angVel(:,1+(j-1)*selL:selL*j,:);

    for i = 1:samples
        temp = angPart(:,:,i)'*angPart(:,:,i);
        Diag(i,j) = sum(diag(temp));
        allDphi(i,j) = 2*sum(temp(mask ~=0));
    end
end


figure
hold on
histogram(Diag)
histogram(allDphi)
hold off
legend('Diagonal','Off diagonal')


yss = Yt(:,:,1);
% figure
% imagesc(yss)


figure
plot(yss')


%% 
% basic parameters
n = 10;              % default 10
k = 3;             % default 3
t = 5e3;              % total number of samples
% step = 10;              % storing every 20 step
% num_store = ceil(t/step);
learnType = 'online';
learnRate = 0.05;
repeat = 1e4;
noiseStd = 5e-2;

eigs = [4.5,3,1.5,0.1*ones(1,7)];

% generate input data
V = orth(randn(n,n));
C = V*diag(eigs)*V';

% generate multivariate Gaussian distribution with specified correlation
% matrix
X = mvnrnd(zeros(1,n),C,t)';

% store the perturbations
record_step = 200;
perturbs = zeros(k,n,round(t/record_step));  %store the perturbation part of F


% Use offline learning to find the inital solution
dt = learnRate; %default 0.005
tau = 0.5;
W = randn(k,n);
M = eye(k,k);
noise1 = noiseStd;
noise2 = noiseStd;
dt0 = 0.05;      % learning rate for the initial phase
tau0 = 0.5;
for i = 1:500
    Y = pinv(M)*W*X; 
    W = (1-2*dt0)*W + 2*dt0*Y*X'/t;
    M = (1-dt0/tau0)*M + dt0/tau0*(Y*Y')/t;
    F = pinv(M)*W;
%     disp(norm(F*F'-eye(k),'fro'))
end


% now add noise to see how representation drift
tot_iter = 1e1;   % total iteration time, longer for better estimation
num_sel = 1;    % only select part of the samples to estimate diffusion costant
step = 1;        % store every 10 step
time_points = round(tot_iter/step);
Yt = zeros(k,time_points,repeat);
inx = randperm(t);
sel_inx = randperm(t,num_sel);

% store all the representations at three time points
Ws = W; Ms = M;
% use online or offline learning

for rp = 1:repeat
    W = Ws;
    M = Ms;  % make sure starting from the same inintial condition
if strcmp(learnType,'offline')
    for i = 1:tot_iter
        % sample noise matrices
        xis = randn(k,n);
        zetas = randn(k,k);
        Y = pinv(M)*W*X; 
        W = (1-2*dt)*W + 2*dt*Y*X'/t + sqrt(2*dt)*noise1*xis;
        M = (1-dt/tau)*M + dt/tau*Y*Y'/t +  sqrt(2*dt)*noise2*zetas;

        if mod(i,step)==0
            Yt(:,round(i/step),rp) = pinv(M)*W*X(:,sel_inx);
        end
    end
elseif strcmp(learnType,'online')
    for i = 1:tot_iter
        % generate noise matrices
        xis = randn(k,n);
        zetas = randn(k,k);
        Y = pinv(M)*W*X(:,inx(i)); 
        W = (1-2*dt)*W + 2*dt*Y*X(:,inx(i))' + sqrt(2*dt)*noise1*xis;
        M = (1-dt/tau)*M + dt/tau*Y*Y' +  sqrt(dt/tau)*noise2*zetas; 
        if mod(i,step)==0
           Yt(:,round(i/step),rp) = pinv(M)*W*X(:,sel_inx);
        end       
    end
end
end

%% 
[k,N,repeat] = size(Yt);
drs = Yt(:,2:end,:) - Yt(:,1:(N-1),:);
lens = sum(Yt(:,2:end,:).^2,1);  % square of vectors
angVel = cross(Yt(:,1:(N-1),:), drs, 1)./lens; % angular velocity

selL = 3;
Num = floor(N/selL)-1;

allDphi = nan(repeat,1);
Diag = nan(repeat,1);
temp = ones(selL);
mask = tril(temp,-1);

for j = 1:repeat
    angPart = angVel(:,1:selL,j);
    
    temp = angPart'*angPart;
    Diag(j) = sum(diag(temp));
    allDphi(j) = 2*sum(temp(mask ~=0));
        
end

figure
hold on
histogram(Diag)
histogram(allDphi)


% example trajectory
ys = Yt(:,:,500);
figure
plot(ys')

sqColor = brewermap(size(angPart,2),'Spectral');
figure
hold on
for i = 1:size(angPart,2)
    plot3(angPart(1,i),angPart(2,i),angPart(3,i),'.','Color',sqColor(i,:))
end
hold off


%% select every 10 data point to estimate the diffusion constant

Yts = Yt(:,1:1:end,:);


[k,N,repeat] = size(Yts);
drs = Yts(:,2:end,:) - Yts(:,1:(N-1),:);
% lens = sum(Yts(:,2:end,:).^2,1);  % square of vectors
lens = sqrt(sum(Yts(:,:,:).^2,1));  % square of vectors
angVel = cross(Yts(:,1:(N-1),:), drs, 1)./lens(:,1:end-1,:)./lens(:,2:end,:); % angular velocity

selL = 1500;
Num = floor(N/selL)-1;

allDphi = nan(repeat,1);
Diag = nan(repeat,1);
temp = ones(selL);
mask = tril(temp,-1);

for j = 1:repeat
    angPart = angVel(:,1:selL,j);
    
    temp = angPart'*angPart;
    Diag(j) = sum(diag(temp));
    allDphi(j) = 2*sum(temp(mask ~=0));
        
end

figure
hold on
histogram(Diag)
histogram(allDphi)


sqColor = brewermap(size(angPart,2),'Spectral');
figure
hold on
for i = 1:size(angPart,2)
    plot3(angPart(1,i),angPart(2,i),angPart(3,i),'o','Color',sqColor(i,:))
end
hold off

figure
plot3(angPart(1,:)',angPart(2,:)',angPart(3,:)')


sqColor = brewermap(size(z,2),'Spectral');
figure
hold on
for i = 1:size(z,2)
    plot3(z(1,i),z(2,i),z(3,i),'.','Color',sqColor(i,:))
end
hold off
