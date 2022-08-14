% This program solves various NSM problems, like eq 43 in the
% Manifold-tiling paper

close all
clear

%% generate input data


%% setup the parameters
param.T = 628;
param.alpha = 0;
param.lambda = 0.045;
param.N = 157;  % number of output neurons

% generate input data
radius = 1;
sep = 2*pi/param.T;
ags =0:sep:2*pi;
X = [radius*cos(sep:sep:2*pi);radius*sin(sep:sep:2*pi)];

param.Q = zeros(param.T);  % initilize the Gram of Y

%% solve the NSM using quadratic programming

% totIter = 1e3;
% y0 = 0.1*cos((1:param.T)'*2*pi/param.T - pi);
y0 = 0.05*cos((1:param.T)'*2*pi/param.T);

y0(y0<0) = 0;
% Y = y0*y0';
% Y =  zeros(param.T, param.T);
Y = zeros(param.T, param.T);
Y(1,:) = y0;
for j = 2:param.T
    Y(j,:) = circshift(y0,j);
end
% Y = Y/sqrt(param.T);

% ys = MantHelper.iterationProjNSM(X,y0,param,0.5/param.T);

Ys = MantHelper.iterationBatchNSM(X,y0,param,1/param.T);
% Ys = MantHelper.quadprogmNSM0(X,Y,param.alpha, param.lambda);

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
figure
plot(Ys(1,:)*sqrt(param.T))
xlabel('$\theta$','Interpreter','latex','FontSize',24)
ylabel('Response','FontSize',24)
set(gca,'LineWidth',1,'FontSize',20,'XTick',[1,314,628],'XTickLabel',{'0','-\pi','\pi'})

%% solve the odes 
% Test ode solver
options = odeset('Vectorized','on');
% y0 = zeros(param.Ne + param.Ni,1);
% x = X(100,:)';
tspan = [0,100];
% [t,y] = ode15s(@(t,y)neuralDynODEs(t,y,param,x),tspan,y0,options);

totIter = 1e4;
y0 = 0.001*rand(param.T,1);
for i = 1:totIter
%     inx = randperm(param.Samp,1);
%     x = X(inx,:)';
    [t,y] = ode15s(@(t,y)NSModes(t,y,param,X),tspan,y0,options);
    ys = y(end,:);
%     param = weightUpdate(param, x,y(end,:)');
    
    Y = zeros(param.T, param.T);
    Y(1,:) = y(end,:)';
    for j = 2:param.T
        Y(j,:) = circshift(ys,j);
    end
    
    % update the Q matrix
    param.Q = Y'*Y;  
end