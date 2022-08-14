% this program simulate the one step update with noise
% modified from driftAnalysis.m


sFolder = './figures';
type = 'psp';           % 'psp' or 'expansion
tau = 0.5;              %  scaling of the learning rate for M
learnType = 'offline';  %  online, offline, batch   

% plot setting
defaultGraphicsSetttings
blues = brewermap(11,'Blues');
rb = brewermap(11,'RdBu');
set1 = brewermap(9,'Set1');



%% learning rate depedent, keep eigen spectrum and noise level, tau the same
n = 10;
k = 3;
% eigens = [4.5,3,2,ones(1,7)*0.01];
eigens = [5,3.5,1.3,ones(1,7)*0.05];
noiseStd = 0.005; % 0.005 for expansion, 0.1 for psp
learnRate = 0.1;

t = 10000;  % total number of iterations

% generate input data, first covariance matrix
V = orth(randn(n,n));
U = orth(randn(k,k));       % arbitrary orthogonal matrix to determine the projection
C = V*diag(eigens)*V'/n;    % with normalized length, Aug 6, 2020

% generate multivariate Gaussian distribution with specified correlation
% matrix
X = mvnrnd(zeros(1,n),C,t)';
Cx = X*X'/t;      % input sample covariance matrix

% generate some test stimuli that are not used during learning
num_test = 1e2;
Xtest = mvnrnd(zeros(1,n),C,num_test)';

% ========================================================
% Use offline learning to find the inital solution
% ========================================================
dt = learnRate;
W = randn(k,n);
M = eye(k,k);
noise1 = noiseStd;
noise2 = noiseStd;
dt0 = 0.1; % learning rate for the initial phase, can be larger for faster convergence
tau0 = 0.5;
for i = 1:100
    Y = pinv(M)*W*X; 
    W = (1-2*dt0)*W + 2*dt0*Y*X'/t;
    M = (1-dt0/tau0)*M + dt0/tau0*(Y*Y')/t;
    F = pinv(M)*W;
    disp(norm(F*F'-eye(k),'fro'))
end

% store the parameters, as a reference parameter set
Ws = W;
Ms = M;
Fs = F;
%% now add noise but only one step

tot_iter = 1e4;     % total number of updates
subPopu = 100;             % number of subpopulation chosen
ensembStep = 1e4;           % store all the representation every ensembStep
partial_index = randperm(t,subPopu); 
Y_ensemble = zeros(k,subPopu,tot_iter);    %store a subpopulation
Y0 = pinv(Ms)*Ws*X(:,partial_index);

if strcmp(learnType,'offline')
    for i = 1:tot_iter
        % sample noise matrices
        xis = randn(k,n);
        zetas = randn(k,k);
        Y = pinv(Ms)*Ws*X; 
        W = (1-2*dt)*Ws + 2*dt*Y*X'/t + sqrt(2*dt)*noise1*xis;
        M = (1-dt/tau)*Ms + dt/tau*Y*Y'/t +  sqrt(dt/tau)*noise2*zetas;

        Y_ensemble(:,:,i) = pinv(M)*W*X(:,partial_index);
    end
elseif strcmp(learnType,'online')
    for i = 1:tot_iter
        curr_inx = randperm(t,1);  % randm generate one sample
        % generate noise matrices
        xis = randn(k,n);
        zetas = randn(k,k);
        Y = pinv(Ms)*Ws*X(:,curr_inx); 
        W = (1-2*dt)*Ws + 2*dt*Y*X(:,curr_inx)' + sqrt(dt)*noise1*xis;
        M = (1-dt/tau)*Ms + dt/tau*Y*Y' +  sqrt(dt)*noise2*zetas; 
        Y_ensemble(:,:,round(i/1e4)+1) = pinv(M)*W*X(:,partial_index);
    end
end


%% Analysis
dY = Y_ensemble - Y0;
% 3d plot showing the distribution after one step perturbation
expY = squeeze(dY(:,4,:));
figure
scatterHd = plot3(expY(1,:),expY(2,:), expY(3,:),'.','MarkerSize',6);
xlabel('$y_1$','Interpreter','latex','FontSize',20)
ylabel('$y_2$','Interpreter','latex','FontSize',20)
zlabel('$y_3$','Interpreter','latex','FontSize',20)
xlim([-0.1,0.1])
ylim([-0.1,0.1])
zlim([-0.1,0.1])
set(gca,'FontSize',16)
grid on

% The PCA of output space

Ys = pinv(Ms)*Ws*X(:,randperm(t,1e4));
[evs,scores,~,~,explained] = pca(Ys');

orthPCs = nan(k,k,subPopu);  % store the orthogonality
PCvar = nan(k,subPopu);  % store the variance explained for individual samples
allPC =  nan(k,k,subPopu); % all eigenvectors derived from all samples
for i = 1:size(dY,2)
    expY = squeeze(dY(:,i,:));
    [coefs,~,~,~,expls] = pca(expY');
    allPC(:,:,i) = coefs;
    orthPCs(:,:,i) = coefs'*evs;
    PCvar(:,i) = expls;
end

% average orthogonality
aveOrthPC = mean(orthPCs,3);
avePertPC = mean(allPC,3);
avePCexpl = mean(PCvar,2);
% compare the output eigenvectors and the perturbation eigenvectors
figure
hold on
for i = 1:k
    plot3([0;evs(1,i)],[0;evs(2,i)],[0;evs(3,i)],'LineWidth',2,'Color',set1(1,:))
    plot3([0;avePertPC(1,i)],[0;avePertPC(2,i)],[0;avePertPC(3,i)],'LineWidth',2,'Color',set1(2,:))
end
hold off


% PCA of the one step perturbation
[coefs,~,~,~,expls] = pca(expY');
sum(evs.*coefs,1)

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
