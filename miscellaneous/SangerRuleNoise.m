tau = 0.5;              %  scaling of the learning rate for M

n = 10;
k = 3;
eigens = [4.5,3.5,1,ones(1,7)*0.01];
noiseStd = 1e-2; % 0.005 for expansion, 0.1 for psp
learnRate = 0.02;

t = 2e4;  % total number of iterations

% generate input data, first covariance matrix
V = orth(randn(n,n));
U = orth(randn(k,k));       % arbitrary orthogonal matrix to determine the projection
C = V*diag(eigens)*V'/sqrt(n);    % with normalized length, Aug 6, 2020
Vnorm = norm(V(:,1:k));

% generate multivariate Gaussian distribution with specified correlation
% matrix
X = mvnrnd(zeros(1,n),C,t)';
Cx = X*X'/t;      % input sample covariance matrix

num_test = 1e2;
Xtest = mvnrnd(zeros(1,n),C,num_test)';


dt = learnRate;
W = randn(k,n);
% M = eye(k,k);

% Add a mask to M in each step
temp = ones(k);
mask = tril(temp);

% F = pinv(M)*W; % inital feature map
noise1 = noiseStd;
% noise2 = noiseStd;
dt0 = 0.02;          % learning rate for the initial phase, can be larger for faster convergence
tau0 = 0.5;
pspErrPre = nan(100,1); % store the psp error during the pre-noise stage
for i = 1:500
    Y = W*X; 
    W = W + dt0*(Y*X'/t - tril(Y*Y'/t)*W);
end

% PCA target project
Zpca = V(:,1:k)'*X;


figure
plot(Zpca(1,:),Y(1,:),'.')


%% Continuous with noise
tot_iter = 5e4;     % total number of updates
num_sel = 200;      % randomly selected samples used to calculate the drift and diffusion constants
step = 10;          % store every 10 updates
time_points = round(tot_iter/step);
Yt = zeros(k,time_points,num_sel);
sel_inx = randperm(t,num_sel);    % samples indices

Zsel = Zpca(:,sel_inx);
% Ytest = zeros(k,time_points,num_test);  % store the testing stimulus


% store all the representations at three time points
subPopu = 1000;             % number of subpopulation chosen
ensembStep = 1e4;           % store all the representation every ensembStep
partial_index = randperm(t,subPopu);   
Y_ensemble = zeros(k,subPopu,round(tot_iter/ensembStep) + 1);    %store a subpopulation

for i = 1:tot_iter
    curr_inx = randperm(t,1);  % randm generate one sample
    % generate noise matrices
    xis = randn(k,n);
    Y = W*X(:,curr_inx); 
    W =  W + dt*(Y*X(:,curr_inx)' - tril(Y*Y')*W) + sqrt(dt)*noise1*xis;

    if mod(i,step)==0
%         temp = pinv(M)*W;  % current feature map
        Yt(:,round(i/step),:) = W*X(:,sel_inx);
    end

    % store population representations every 10000 steps
    if mod(i, ensembStep)==0 || i == 1
       Y_ensemble(:,:,round(i/1e4)+1) = W*X(:,partial_index);
    end     
end


% Make the plot more pretty

% plot the figure
pointGap = 20;
xs = (1:pointGap:time_points)'*step;
selYInx = randperm(num_sel,1);
ys = Yt(:,:,selYInx);

figure
hold on
for i = 1:k
    plot(xs,ys(i,1:pointGap:time_points)','LineWidth',2)
    plot(xs,ones(length(xs),1)*Zsel(i,selYInx),'k--')
end
hold off

box on
bLgh = legend('y_1','y_2','y_3','Location','northwest');
legend boxoff
set(bLgh,'FontSize',14)
xlabel('$t$','Interpreter','latex','FontSize',20)
ylabel('$y_i(t)$','Interpreter','latex','FontSize',20)
set(gca,'LineWidth',1,'FontSize',16)
