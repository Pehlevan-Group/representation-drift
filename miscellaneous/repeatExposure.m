% this program tests the idea that repeat expoure might stabalize the
% representation

clear all;
close all;

n = 3; % default 3
k = 10;
t = 20000;
step = 4; %storing every 20 step
num_store = ceil(t/step);

V = orth(randn(n,n));
% C = V*diag([5,3,1.3,0.1*ones(1,7)])*V';
C = V*diag([2 1 0.1])*V';
X = mvnrnd(zeros(1,n),C,t)';

% select one sample to repeat
sel_sample = randperm(t,1); % selected index
num_repeat = 1000;
step_repeat = 40;
repr_sel = nan(k,floor(t/step_repeat));
% repeat_type = 'continous'; % or seperate
step_intro = 15000;  % step to introduce the odors

dt = 0.01; %default 0.005
tf = t;

noise = 0.01;  % default 0.02
tau = 0.5;

W = randn(k,n);
M = eye(k,k);
%Y = pinv(M)*W*X;
sel = randperm(t,10);
Yt = zeros(k,num_store,10);
% Zt = zeros(k,tf);
E = zeros(1,tf);

count = 1;
for i = 1:tf
    
    ind = randperm(t,1);
    
    Y = pinv(M)*W*X(:,ind);
%     Y = nonNegativeProjection(X(:,ind),W,M);
    W = (1-2*dt)*W + 2*dt*(Y*X(:,ind)'+noise*randn(k,n));
    M = (1-dt/tau)*M + dt/tau*(Y*Y'+noise*randn(k,k));

%     every step_repeat, present the testing sample
    if mod(i,step_repeat)==0
        % update when exposed to this stimulus
        Y = pinv(M)*W*X(:,sel_sample);
%         Y = nonNegativeProjection(X(:,sel_sample),W,M);
        W = (1-2*dt)*W + 2*dt*(Y*X(:,sel_sample)'+noise*randn(k,n));
        M = (1-dt/tau)*M + dt/tau*(Y*Y'+noise*randn(k,k));
        % check the representation
%         repr_sel(:,count) = pinv(M)*W*X(:,sel_sample);
        repr_sel(:,count) = nonNegativeProjection(X(:,sel_sample),W,M);
        count = count +1;
    end
    
    % introduce the repeat odor
%     if i == step_intro
%         for rp = 1:num_repeat
% %             Y = pinv(M)*W*X(:,sel_sample); 
%             Y = nonNegativeProjection(X(:,sel_sample),W,M);
%             W = (1-2*dt)*W + 2*dt*(Y*X(:,sel_sample)'+noise*randn(k,n));
%             M = (1-dt/tau)*M + dt/tau*(Y*Y'+noise*randn(k,k));
%             % check the representation
% %             repr_sel(:,rp) = pinv(M)*W*X(:,sel_sample);
%             repr_sel(:,rp) = nonNegativeProjection(X(:,sel_sample),W,M);
%         end
%     end
        
    F = pinv(M)*W;
    if mod(i,step)==1
        Ft(ceil(i/step),:,:) = F;
    end
    E(1,i) = norm(F*F'-eye(k),'fro');
    SE(1,i) = norm(F'*F-V(:,1:2)*V(:,1:2)','fro');
    if mod(i,step)==1
        for j = 1:10
            Yt(:,ceil(i/step),j) = pinv(M)*W*X(:,j);
%             Yt(:,ceil(i/step),j) = activationFun(X(:,j),W,M,'relu');
        end   
    end
end

% the representation of the target sample
set1 = brewermap(7,'Set1');
min(7,k)
figure
for i = 1:min(7,k)
    plot(1:size(repr_sel,2),repr_sel(i,:),'LineWidth',1.5,'Color',set1(i,:));
    hold on
end
hold off
xlabel('repeats','FontSize',28)
ylabel('$y_i(t)$','Interpreter','latex','FontSize',28)
set(gca,'FontSize',24,'LineWidth',1.5)

a = Ft(100:end,:);
figure
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
min(7,k)
figure
for i = 1:min(7,k)
    plot(step*(100:ceil(tf/step))',Yt(i,100:end,6),'LineWidth',1.5,'Color',set1(i,:));
    hold on
end
hold off
xlabel('T','FontSize',28)
ylabel('$y_i(t)$','Interpreter','latex','FontSize',28)
set(gca,'FontSize',24,'LineWidth',1.5)

% population representation
sel_inx = [1000,2500,5000];
figure
for i = 1:length(sel_inx)
    subplot(1,3,i)
%     imagesc(squeeze(Yt(:,sel_inx(i),:)),[-1.5,1.5]);colorbar
    imagesc(squeeze(Yt(:,sel_inx(i),:)));colorbar
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
        for l0 = 1:length(ix)
            sm(i,j,l0) = pdist([Yt(:,ix(l0),i)';Yt(:,ix(l0),j)'],'cosine');
%             sm(i,j,l0) = Yt(:,ix(l0),i)'*Yt(:,ix(l0),j);
        end
    end
end

S1 = sm(:,:,50);S2 = sm(:,:,100);

figure

figureSize = [0 0 12 7];
set(gcf,'Units','inches','Position',figureSize,'PaperPositionMode','auto');

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

