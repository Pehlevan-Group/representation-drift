% This program demonstrate that degeneracy of the target function is
% necessary to generate drifting behavior
% Compare PSP, PSP with M matrix symmetry breaking, original Sanger's rule
% with only forward matrix W

% We show that representation of the same input X and how it change over
% time

% Simulation data used for ploting Figure S8
close all
clc

%% define some colors
blues = brewermap(11,'Blues');
BuGns = brewermap(11,'BuGn');
greens = brewermap(11,'Greens');
greys = brewermap(11,'Greys');
PuRd = brewermap(11,'PuRd');
YlOrRd = brewermap(11,'YlOrRd');
oranges = brewermap(11,'Oranges');

%% Parameters
sFolder = './figures';
type = 'psp';           % 'psp' or 'expansion
tau = 0.5;              %  scaling of the learning rate for M
learnType = 'online';   %  online, offline, batch   

n = 10;
k = 3;
eigens = [4.5,3.5,1,ones(1,7)*0.01];
noiseStd = 1e-2; % 0.005 for expansion, 0.1 for psp
learnRate = 0.01;

t = 1e4;  % total number of iterations

% generate input data, first covariance matrix
V = orth(randn(n,n));
U = orth(randn(k,k));       % arbitrary orthogonal matrix to determine the projection
C = V*diag(eigens)*V'/sqrt(n);    % with normalized length, Aug 6, 2020
Vnorm = norm(V(:,1:k));

% generate multivariate Gaussian distribution with specified correlation
% matrix
X = mvnrnd(zeros(1,n),C,t)';
Cx = X*X'/t;      % input sample covarian

% select one input to track
xsel = X(:,randperm(t,1)); 

%% PSP with M matrix symmetry-breaked
% ========================================================
% Use offline learning to find the inital solution
% ========================================================

dt = learnRate;
W = randn(k,n);
M = eye(k,k);

% Add a mask to M in each step
temp = ones(k);
mask = tril(temp);

% F = pinv(M)*W; % inital feature map
noise1 = noiseStd;
noise2 = noiseStd;
dt0 = 0.1;          % learning rate for the initial phase, can be larger for faster convergence
tau0 = 0.5;
pspErrPre = nan(100,1); % store the psp error during the pre-noise stage
for i = 1:100
    Y = pinv(M)*W*X; 
    W = (1-2*dt0)*W + 2*dt0*Y*X'/t + sqrt(2*dt)*noise1*randn(k,n);
    M = (1-dt0/tau0)*M + dt0/tau0*(Y*Y')/t;
    M = M.*mask;
    F = pinv(M)*W;
    disp(norm(F*F'-eye(k),'fro'))
%     pspErrPre(i) = norm(F'*F - V(:,1:k)*V(:,1:k)');
end

% ========================================================
% online learning with synaptic noise
% ========================================================


tot_iter = 2e5;     % total number of updates
step = 20;          % store every 10 updates
time_points = round(tot_iter/step);
Yt_sb = zeros(k,time_points);
pspErr = nan(time_points,1);
% Ytest = zeros(k,time_points,num_test);  % store the testing stimulus

Ft = zeros(k,n,time_points);        % store the feature maps
inx = randperm(t);
% sel_inx = randperm(t,1);    % selct on

for i = 1:tot_iter
    curr_inx = randperm(t,1);  % randm generate one sample
    % generate noise matrices
    xis = randn(k,n);
    zetas = randn(k,k);
    Y = pinv(M)*W*X(:,curr_inx); 
    W = (1-2*dt)*W + 2*dt*Y*X(:,curr_inx)' + sqrt(dt)*noise1*xis;
    M = (1-dt/tau)*M + dt/tau*Y*Y' +  sqrt(dt)*noise2*zetas; 
    M = M.*mask;

    if mod(i,step)==0
        temp = pinv(M)*W;       % current feature map
        Yt_sb(:,round(i/step)) = temp*xsel;

            % PSP error
        if strcmp(type,'psp')
            pspErr(round(i/step)) = norm(temp'*temp - V(:,1:k)*V(:,1:k)');
        end
    end  
end


% *********************************************
% Plot the figure, with publication quality
% *********************************************
fh_sb = figure;
set(fh_sb,'color','w','Units','inches')
pos(3)=4;  
pos(4)=3;
set(gcf,'Position',pos)
labelFontSize = 24;
gcaFontSize = 20;

colorsSel = [YlOrRd(6,:);PuRd(8,:);blues(8,:)];

pointGap = 20;
xs = (1:pointGap:time_points)'*step;
for i = 1:k
    plot(xs,Yt_sb(i,1:pointGap:time_points)','Color',colorsSel(i,:),'LineWidth',2)
    hold on
end
hold off
bLgh = legend('y_1','y_2','y_3','Location','northwest');
legend boxoff
set(bLgh,'FontSize',14)
xlabel('$t$','Interpreter','latex','FontSize',labelFontSize)
ylabel('$y_i(t)$','Interpreter','latex','FontSize',labelFontSize)
set(gca,'LineWidth',1,'FontSize',gcaFontSize)

prefix = 'noisy_PSP_M_symm_break';
saveas(fh_sb,[sFolder,filesep,prefix,'.fig'])
print('-depsc',[sFolder,filesep,prefix,'.eps'])


dotColors = flip(brewermap(size(Yt_sb,2)/pointGap,'Spectral'));
figure
hold on
for i = 1:(size(Yt_sb,2)/pointGap)
    plot3(Yt_sb(1,i*pointGap),Yt_sb(2,i*pointGap),Yt_sb(3,i*pointGap),...
        '.','MarkerSize',8,'Color',dotColors(i,:))
end



%% PSP

% ========================================================
% Use offline learning to find the inital solution
% ========================================================

dt = learnRate;
W_psp = randn(k,n);
M_psp = eye(k,k);


% F = pinv(M)*W; % inital feature map
noise1 = noiseStd;
noise2 = noiseStd;
dt0 = 0.1;          % learning rate for the initial phase, can be larger for faster convergence
tau0 = 0.5;
pspErrPre = nan(100,1); % store the psp error during the pre-noise stage
for i = 1:100
    Y = pinv(M_psp)*W_psp*X; 
    W_psp = (1-2*dt0)*W_psp + 2*dt0*Y*X'/t + sqrt(2*dt)*noise1*randn(k,n);
    M_psp = (1-dt0/tau0)*M_psp + dt0/tau0*(Y*Y')/t;
    F_psp = pinv(M_psp)*W_psp;
    disp(norm(F_psp*F_psp'-eye(k),'fro'))
%     pspErrPre(i) = norm(F'*F - V(:,1:k)*V(:,1:k)');
end

% ========================================================
% online learning with synaptic noise
% ========================================================


time_points = round(tot_iter/step);
Yt_psp = zeros(k,time_points);
pspErr = nan(time_points,1);
% Ytest = zeros(k,time_points,num_test);  % store the testing stimulus

for i = 1:tot_iter
    curr_inx = randperm(t,1);  % randm generate one sample
    % generate noise matrices
    xis = randn(k,n);
    zetas = randn(k,k);
    Y = pinv(M_psp)*W_psp*X(:,curr_inx); 
    W_psp = (1-2*dt)*W_psp + 2*dt*Y*X(:,curr_inx)' + sqrt(dt)*noise1*xis;
    M_psp = (1-dt/tau)*M_psp + dt/tau*Y*Y' +  sqrt(dt)*noise2*zetas; 

    if mod(i,step)==0
        temp = pinv(M_psp)*W_psp;       % current feature map
        Yt_psp(:,round(i/step)) = temp*xsel;

            % PSP error
        if strcmp(type,'psp')
            pspErr(round(i/step)) = norm(temp'*temp - V(:,1:k)*V(:,1:k)');
        end
    end  
end


% *********************************************
% Plot the figure, with publication quality
% *********************************************
fh_psp = figure;
set(fh_psp,'color','w','Units','inches')
pos(3)=4;  
pos(4)=3;
set(gcf,'Position',pos)
labelFontSize = 24;
gcaFontSize = 20;

colorsSel = [YlOrRd(6,:);PuRd(8,:);blues(8,:)];

% pointGap = 5;
xs = (1:pointGap:time_points)'*step;
for i = 1:k
    plot(xs,Yt_psp(i,1:pointGap:time_points)','Color',colorsSel(i,:),'LineWidth',2)
    hold on
end
hold off
bLgh = legend('y_1','y_2','y_3','Location','northwest');
legend boxoff
set(bLgh,'FontSize',14)
xlabel('$t$','Interpreter','latex','FontSize',labelFontSize)
ylabel('$y_i(t)$','Interpreter','latex','FontSize',labelFontSize)
set(gca,'LineWidth',1,'FontSize',gcaFontSize)

prefix = 'noisy_PSP_compare';
saveas(fh_psp,[sFolder,filesep,prefix,'.fig'])
print('-depsc',[sFolder,filesep,prefix,'.eps'])

dotColors = flip(brewermap(size(Yt_psp,2)/pointGap,'Spectral'));
figure
hold on
for i = 1:(size(Yt_psp,2)/pointGap)
    plot3(Yt_psp(1,i*pointGap),Yt_psp(2,i*pointGap),Yt_psp(3,i*pointGap),...
        '.','MarkerSize',8,'Color',dotColors(i,:))
end


%% Sanger's learning rule with only forward matrix

Ws = randn(k,n);

% Add a mask to M in each step
temp = ones(k);
mask = tril(temp);

% F = pinv(M)*W; % inital feature map
noise1 = noiseStd;
% noise2 = noiseStd;
dt0 = 0.05;          % learning rate for the initial phase, can be larger for faster convergence
tau0 = 0.5;
pspErrPre = nan(100,1); % store the psp error during the pre-noise stage
for i = 1:500
    Y = Ws*X; 
    Ws = Ws + dt0*(Y*X'/t - tril(Y*Y'/t)*Ws);
end


% PCA target project
Zpca = V(:,1:k)'*X;


figure
plot(Zpca(1,:),Y(1,:),'.')

Yt_sg = zeros(k,time_points);

for i = 1:tot_iter
    curr_inx = randperm(t,1);  % randm generate one sample
    % generate noise matrices
    xis = randn(k,n);
    Y = Ws*X(:,curr_inx); 
    Ws =  Ws + dt*(Y*X(:,curr_inx)' - tril(Y*Y')*Ws) + sqrt(dt)*noise1*xis;

    if mod(i,step)==0
        Yt_sg(:,round(i/step),:) = Ws*xsel;
    end
end


% *********************************************
% Plot the figure, with publication quality
% *********************************************
fh_sg = figure;
set(fh_sg,'color','w','Units','inches')
pos(3)=4;  
pos(4)=3;
set(fh_sg,'Position',pos)
labelFontSize = 24;
gcaFontSize = 20;

% colorsSel = [YlOrRd(6,:);PuRd(8,:);blues(8,:)];

% pointGap = 5;
% xs = (1:pointGap:time_points)'*step;
for i = 1:k
    plot(xs,Yt_sg(i,1:pointGap:time_points)','Color',colorsSel(i,:),'LineWidth',2)
    hold on
end
hold off
bLgh = legend('y_1','y_2','y_3','Location','northwest');
legend boxoff
set(bLgh,'FontSize',14)
xlabel('$t$','Interpreter','latex','FontSize',labelFontSize)
ylabel('$y_i(t)$','Interpreter','latex','FontSize',labelFontSize)
set(gca,'LineWidth',1,'FontSize',gcaFontSize)

prefix = 'noisy_Sanger';
saveas(fh_sg,[sFolder,filesep,prefix,'.fig'])
print('-depsc',[sFolder,filesep,prefix,'.eps'])


dotColors = flip(brewermap(size(Yt_sg,2)/pointGap,'Spectral'));
figure
hold on
for i = 1:(size(Yt_sg,2)/pointGap)
    plot3(Yt_sg(1,i*pointGap),Yt_sg(2,i*pointGap),Yt_sg(3,i*pointGap),...
        '.','MarkerSize',8,'Color',dotColors(i,:))
end
