
% toy model to test how the feature map drift due to Hebbian/anti-Hebbian
% learning

clear all;
close all;

n = 10; % default 3
k = 3;
t = 20000;
step = 20; %storing every 20 step
num_store = ceil(t/step);

V = orth(randn(n,n));
U = orth(randn(k,k));  % arbitrary orthogonal matrix to determine the projection
% C = V*diag([2,0.9,0.1])*V';
C = V*diag([5,3,1.3,0.1*ones(1,7)])*V';

% C = V*diag([5,3,1.5,0.05*ones(1,7)])*V';
% C = V*diag([5,3,1.3,0.01*ones(1,7)])*V';

% C = V*diag([2 1 0.1])*V';
X = mvnrnd(zeros(1,n),C,t)';

% one of the PSP
% [ev,eigens] = eigs(X*X'/t);
% eig_vec = X'*ev(:,1:k);  % only keep the largest k
% part_diag = diag(sqrt(diag(eigens(1:k,:))));
% Y = U * part_diag * eig_vec';
% 
% % define the solution of W and M
% W = Y*X'/t;
% M = Y*Y'/t;

% feature
% F = M\W;
% offline error
% erf_off = norm(X'*X/t- Y'*Y/t,'fro');

% Use offline learning
dt = 0.01; %default 0.005
tau = 0.5;
W = randn(k,n);
M = eye(k,k);
noise = 1e-4;
for i = 1:100
    Y = pinv(M)*W*X; 
    W = (1-2*dt)*W + 2*dt*Y*X'/t;
    M = (1-dt/tau)*M + dt/tau*(Y*Y')/t;
    F = pinv(M)*W;
    disp(norm(F*F'-eye(k),'fro'))
end

% now add noise to see how F drift
tot_iter = 20000;
num_sel = 100;
step = 10;
time_points = round(tot_iter/step);
Yt = zeros(k,time_points,num_sel);
inx = randperm(t);
sel_inx = randperm(t,num_sel);
for i = 1:tot_iter
%     Y = pinv(M)*W*X; 
%     W = (1-2*dt)*W + 2*dt*(Y*X'/t + noise*randn(k,n));
%     M = (1-dt/tau)*M + dt/tau*(Y*Y'/t +  noise*randn(k,k));
    
    Y = pinv(M)*W*X(:,i); 
%     W = (1-2*dt)*W + 2*dt*(Y*X(:,i)'/t + noise*randn(k,n));
    W = (1-2*dt)*W + 2*dt*(Y*X(:,i)'/t);
    M = (1-dt/tau)*M + dt/tau*(Y*Y'/t +  noise*randn(k,k));
%     W = W + 2*dt*(noise*randn(k,n));
%     M = M + dt/tau*noise*randn(k,k);
    if mod(i,step)==0
        Ft(:,:,round(i/step)) = pinv(M)*W;
        Yt(:,round(i/step),:) = pinv(M)*W*X(:,sel_inx);
    end
    
end

% representation
figure
subplot(3,1,1)
imagesc(squeeze(Yt(:,1,:))); colorbar
subplot(3,1,2)
imagesc(squeeze(Yt(:,50,:)));colorbar
subplot(3,1,3)
imagesc(squeeze(Yt(:,100,:)));colorbar

% trajectory
figure
plot(squeeze(Yt(1,:,1)),squeeze(Yt(2,:,1)))

% similarity
figure
S1 = squeeze(Yt(:,1,:));S2= squeeze(Yt(:,100,:));
subplot(1,2,1)
imagesc(S1'*S1);colorbar
subplot(1,2,2)
imagesc(S2'*S2);colorbar

% drift of feature map
figure
Ft1=squeeze(Ft(:,:,1));Ft2 = squeeze(Ft(:,:,50));Ft3 = squeeze(Ft(:,:,100));
subplot(1,3,1)
imagesc(Ft1*Ft1'); colorbar
subplot(1,3,2)
imagesc(Ft2*Ft2');colorbar
subplot(1,3,3)
imagesc(Ft3*Ft3');colorbar

% Ft
figure
subplot(1,3,1)
imagesc(Ft1); colorbar
subplot(1,3,2)
imagesc(Ft2);colorbar
subplot(1,3,3)
imagesc(Ft3);colorbar

% distribution of the two trajectory
Y1 = Yt(:,:,1); Y2 = Yt(:,:,4);
figure
plot3(Y1(1,:),Y1(2,:),Y1(3,:),'.')
hold on
plot3(Y2(1,:),Y2(2,:),Y2(3,:),'.')
xlabel('$y_1$','Interpreter','latex','FontSize',24)
ylabel('$y_2$','Interpreter','latex','FontSize',24)
zlabel('$y_3$','Interpreter','latex','FontSize',24)
grid on
set(gca,'FontSize',20)

% projection of the ensemble
Y =  pinv(M)*W*X;
figure
plot3(Y(1,1:10:end),Y(2,1:10:end),Y(3,1:10:end),'.')
xlabel('$y_1$','Interpreter','latex','FontSize',24)
ylabel('$y_2$','Interpreter','latex','FontSize',24)
zlabel('$y_3$','Interpreter','latex','FontSize',24)
grid on
set(gca,'FontSize',20)


% distribution of inner product
d1 = sqrt(sum(Y1.*Y1,1));d2 = sqrt(sum(Y2.*Y2,1));
figure
histogram(d1,30)
hold on
histogram(d2,30)
hold off
legend('sample 1','sample 2')
xlabel('radius','FontSize',28)
ylabel('count','FontSize',28)
set(gca,'FontSize',24,'LineWidth',1.5)

% pairwise similarity
ps = sum(Y1.*Y2,1)./(d1.*d2);
figure
plot((1:length(ps))*10,ps,'LineWidth',2)
xlabel('iterations','FontSize',28)
ylabel('cosine similarity','FontSize',28)
set(gca,'FontSize',24,'LineWidth',1.5,'YLim',[0,1])

figure
histogram(ps)


% calculate the diffusion constants using mean squre displacement
ds = diff(Yt,1,2);
displace = squeeze(sum(ds.*ds,1));
msd = cumsum(displace,1);
plot((1:1999)',msd)

% normalize msd to its radius
radi = mean(squeeze(sum(Yt.*Yt,1)),1);
norm_msd = msd./(ones(size(msd,1),1)*radi);
plot((1:1999)',norm_msd)

% linear regression
x0 = (1:1999)'*dt*step;

b = x0\mean(norm_msd,2);

% random rotation of the feature map
iner_prod = squeeze(sum(Yt(:,1:end-1,:).*Yt(:,2:end,:),1));
lens = sqrt(squeeze(sum(Yt.*Yt,1)));
dtheta = acos(iner_prod./(lens(1:end-1,:).*lens(2:end,:)));
msd_dtheta = cumsum(dtheta,1);
plot((1:1999)',msd_dtheta)
%%

dt = 0.001; %default 0.005
tf = t;

noise = 0.05;
tau = 0.5;

W = randn(k,n);
M = eye(k,k);
%Y = pinv(M)*W*X;
sel = randperm(t,10);
Yt = zeros(k,num_store,10);
% Zt = zeros(k,tf);
E = zeros(1,tf);

for i = 1:tf
    
    ind = randperm(t,1);
    
    Y = pinv(M)*W*X(:,ind); 
    W = (1-2*dt)*W + 2*dt*(Y*X(:,ind)'+noise*randn(k,n));
    M = (1-dt/tau)*M + dt/tau*(Y*Y'+noise*randn(k,k));

    F = pinv(M)*W;
    if mod(i,step)==1
        Ft(ceil(i/step),:,:) = F;
    end
    E(1,i) = norm(F*F'-eye(k),'fro');
    SE(1,i) = norm(F'*F-V(:,1:2)*V(:,1:2)','fro');
    if mod(i,step)==1
        for j = 1:10
            Yt(:,ceil(i/step),j) = pinv(M)*W*X(:,j);
        end   
    end
end

a = Ft(100:end,:);
imagesc(cov(a')); colorbar;
title('Cov(Ft)')
xlabel('time','FontSize',24)
ylabel('time','FontSize',24)
set(gca,'FontSize',20,'LineWidth',1.5)



figure
semilogy(SE)

figure
semilogy(E)


% set1 = brewermap(7,'Set1');
% temporal profile
set1 = brewermap(7,'Set1');
figure
for i = 1:7
    plot(step*(100:ceil(tf/step))',Yt(i,100:end,3),'LineWidth',1.5,'Color',set1(i,:));
    hold on
end
hold off
xlabel('T','FontSize',28)
ylabel('$y_i(t)$','Interpreter','latex','FontSize',28)
set(gca,'FontSize',24,'LineWidth',1.5)

% population representation
sel_inx = [100,500,1000];
figure
for i = 1:length(sel_inx)
    subplot(1,3,i)
    imagesc(squeeze(Yt(:,sel_inx(i),:)),[-1.5,1.5]);colorbar
    title(['iteration:',num2str(sel_inx(i)*step)])
    xlabel('Sample index','FontSize',20)
    set(gca,'FontSize',20)
end


% projection trajectory
figure
plot(Yt(1,100:1:end,1), Yt(2,100:1:end,1))
xlabel('$y_1$','Interpreter','latex','FontSize',28)
ylabel('$y_2$','Interpreter','latex','FontSize',28)
set(gca,'FontSize',24,'LineWidth',1.5)

% projection trajectory
figure
plot3(Yt(1,100:1:end,1),Yt(2,100:1:end,1),Yt(3,100:1:end,1),'LineWidth',1)
xlabel('$y_1$','Interpreter','latex','FontSize',24)
ylabel('$y_2$','Interpreter','latex','FontSize',24)
zlabel('$y_3$','Interpreter','latex','FontSize',24)
grid on
set(gca,'FontSize',20)


% pairwie correlation

ix = 1:10:ceil(tf/step);
sm = nan(10,10,length(ix));
for i = 1:10
    for j = 1:10
        for k = 1:length(ix)
            sm(i,j,k) = Yt(:,ix(k),i)'*Yt(:,ix(k),j);
        end
    end
end

S1 = sm(:,:,50);S2 = sm(:,:,100);
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

% figure Feature map
F = squeeze(Ft(end,:,:));
figure
imagesc(F*F');colorbar
title('$FF^{T}$','Interpreter','latex','FontSize',28)
set(gca,'FontSize',24)

figure
imagesc(F'*F);colorbar
title('$F^{T}F$','Interpreter','latex','FontSize',28)
set(gca,'FontSize',24)

% PCA of the output representations
Yend = F * X;
[eigv,SCORE, LATENT, TSQUARED, EXPLAINED] = pca(Yend');


% project all data two the first three component
pc3 = Yend' * eigv(:,1:3);
figure
plot3(pc3(1:10:end,1),pc3(1:10:end,2),pc3(1:10:end,3),'.','LineWidth',1)
xlabel(['pc1: ',num2str(round(EXPLAINED(1)*10)/10),'%'],'FontSize',24)
ylabel(['pc2: ',num2str(round(EXPLAINED(2)*10)/10),'%'],'FontSize',24)
zlabel(['pc3: ',num2str(round(EXPLAINED(3)*10)/10),'%'],'FontSize',24)
set(gca,'XLim',[-5,5],'YLim',[-5,5],'ZLim',[-5,5])
grid on
set(gca,'FontSize',20)

% project a one data sample along training
sel_ix = 1;
psp1 = Yt(:,:,sel_ix)'*eigv(:,1:3);
psp2 = Yt(:,:,2)'*eigv(:,1:3);
figure
plot3(psp1(100:end,1),psp1(100:end,2),psp1(100:end,3),'-','LineWidth',1)
hold on
plot3(psp2(100:end,1),psp2(100:end,2),psp2(100:end,3),'-','LineWidth',1)
hold off
grid on
xlabel('pc1','FontSize',24)
ylabel('pc2','FontSize',24)
zlabel('pc3','FontSize',24)
set(gca,'FontSize',20)

