% This program solves the coupled NSM problems encounted in my project, like eq 43 in the
% Manifold-tiling paper

% close all
% clear

% Uncomment this when running on a cluster
addpath('/n/home09/ssqin/EI_networks')
% dFolder = ['/n/home09/ssqin/EI_networks',filesep,'mnist'];
 
% start the parallel pool with 12 workers
parpool('local', str2num(getenv('SLURM_CPUS_PER_TASK')));


%% generate random combination of four scaling factors
% set the rho_EE = 1 fixed
Ns = 1e3;    % number of samples
s0 = lhsdesign(Ns, 4);
rho_range_log = [-2,2];  % the range of all the parameters, 10^-2 to 10^2
rhos = 10.^(s0*(rho_range_log(2)-rho_range_log(1)) + rho_range_log(1));

%% setup the parameters
param.T = 628;
param.alpha = 0;
param.lambdaE = 0.045;
param.lambdaI = 0.1;

rhoee = 1;    % we set rhoee as the reference point


param.NE = 157;  % number of excitatory neurons
param.NI = 157;  % number of inhibitory neurons

% param.N = 100; % number of output neurons

% generate input data
radius = 1;
sep = 2*pi/param.T;
ags =0:sep:2*pi;
X = [radius*cos(sep:sep:2*pi);radius*sin(sep:sep:2*pi)];

param.Q = zeros(param.T);  % initilize the Gram of Y

%% solve the NSM using quadratic programming

% totIter = 1e3;
ye0 = 1e-3*cos((1:param.T)'*2*pi/param.T )/sqrt(param.NE);
ye0(ye0<0) = 0;

yi0 = 2e-3*cos((1:param.T)'*2*pi/param.T)/sqrt(param.NI);
yi0(yi0<0) = 0;

allYEs = cell(Ns,1); 
allYIs = cell(Ns,1); 
parfor i = 1:Ns
    % define the scaling factors
    rhos_sel = rhos(i,:);
    params = setRhos(param,rhoee,rhos_sel);
    
    % show the current rhos used in the model
    fprintf('Current rho_EE =  %.2f\n',params.rhoee)
    fprintf('Current rho_Ex =  %.2f\n',params.rhoex)
    fprintf('Current rho_Ix =  %.2f\n',params.rhoix)
    fprintf('Current rho_EI =  %.2f\n',params.rhoei)
    fprintf('Current rho_II =  %.2f\n',params.rhoii)

    % ys = MantHelper.iterationProjNSM(X,y0,param,0.2/param.T);
    [YEs,YIs] = MantHelper.iterationBatchEI_NSM(X,ye0,yi0,params,0.1/params.T);
    % Ys = MantHelper.iterationBatchNSM(X,y0,params, 0.2/param.T);
    % store all the simulation resulsts
    allYEs{i} = YEs;
    allYIs{i} = YIs;
end


% save the data
dFile  = ['NSM_EI_solutions_',date,'.mat'];
save(dFile,'-v7.3','param','allYEs','allYIs')

% for i = 1:totIter
% %     ys = MantHelper.quadprogramNSM(X,Y,param);
% %       ys = MantHelper.lsqlinNSM(X,Y,param,y0);
% %     ys = MantHelper.lsqnonlinNSM(X,Y,param,y0);
%     ys = MantHelper.iterationProjNSM(X,Y,param,0.4/param.T);
%     
%     Y = zeros(param.T, param.T);
%     Y(1,:) = ys';
%     for j = 2:param.T
%         Y(j,:) = circshift(ys,j);
%     end
%     
%     % update the Q matrix
% %     param.Q = Y'*Y;  
% end

% Visualize the solution
% Excitatory responses
% figure
% plot(YEs(1,:)*sqrt(param.T))
% xlabel('$\theta$','Interpreter','latex','FontSize',24)
% ylabel('Response','FontSize',24)
% set(gca,'LineWidth',1,'FontSize',20,'XTick',[1,314,628],'XTickLabel',{'0','-\pi','\pi'})
% 
% % Inhibitory responses
% figure
% plot(YIs(1,:)*sqrt(param.T))
% xlabel('$\theta$','Interpreter','latex','FontSize',24)
% ylabel('Response','FontSize',24)
% set(gca,'LineWidth',1,'FontSize',20,'XTick',[1,314,628],'XTickLabel',{'0','-\pi','\pi'})

%% solve the odes 
% Test ode solver
% options = odeset('Vectorized','on');
% % y0 = zeros(param.Ne + param.Ni,1);
% % x = X(100,:)';
% tspan = [0,100];
% % [t,y] = ode15s(@(t,y)neuralDynODEs(t,y,param,x),tspan,y0,options);
% 
% totIter = 1e4;
% y0 = 0.001*rand(param.T,1);
% for i = 1:totIter
% %     inx = randperm(param.Samp,1);
% %     x = X(inx,:)';
%     [t,y] = ode15s(@(t,y)NSModes(t,y,param,X),tspan,y0,options);
%     ys = y(end,:);
% %     param = weightUpdate(param, x,y(end,:)');
%     
%     Y = zeros(param.T, param.T);
%     Y(1,:) = y(end,:)';
%     for j = 2:param.T
%         Y(j,:) = circshift(ys,j);
%     end
%     
%     % update the Q matrix
%     param.Q = Y'*Y;  
% end