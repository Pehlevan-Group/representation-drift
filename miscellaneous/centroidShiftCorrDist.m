% plot the distance dependent correlation of centroid shift in the ring
% model. Compare input noise and synatpic noise
% Derivation can be found in the supplementary material of the mansucript


%% Default settings of graphics

sFolder = './figures';

blues = brewermap(11,'Blues');
reds = brewermap(11,'Reds');
greys = brewermap(11,'Greys');

% the range of disntace
dphis = 0:0.02:pi;
mu = 1;   % amplitude of RF
%% synaptic noise only scenario

eta = 0.01;  % fixed the learning rate
sigs = [1e-3,1e-2,0.05,0.1];  % change the 

dphi_Corr = @(x,sig) ((1-x/pi).*cos(2*x) + 1/2/pi*sin(2*x))./(1+16*sig^2/mu^2/eta);
figure
hold on
for i = 1:length(sigs)
    dp = dphi_Corr(dphis,sigs(i));
    plot(dphis,dp,'LineWidth',3,'Color',blues(2+2*i,:))
end
plot([0;pi],[0;0],'--','LineWidth',1.5,'Color',greys(7,:))
hold off
xlabel("$|\phi - \phi'|$",'Interpreter','latex')
ylabel('$\rho$','Interpreter','latex')
box on
legend('\sigma = 10^{-3}','\sigma = 10^{-2}','\sigma = 0.05','\sigma = 0.1')
title(['$\eta = ',num2str(eta),'$'],'Interpreter','latex')


%% Input noise only scenario

eta = 0.01;  % fixed the learning rate
sigs = [1e-3,1e-2,0.05,0.1];  % change the 

dphi_Corr_input = @(x,sig) ((1-x/pi).*cos(2*x) + sin(2*x)/pi - ...
    4*sigs(i)^2*((1-x/pi).*cos(x) + sin(x)/pi))./(1+4*sig^2);
figure
hold on
for i = 1:length(sigs)
    dp = dphi_Corr_input(dphis,sigs(i));
    plot(dphis,dp,'LineWidth',3,'Color',blues(2+2*i,:))
end
plot([0;pi],[0;0],'--','LineWidth',1.5,'Color',greys(7,:))
hold off
xlabel("$|\phi - \phi'|$",'Interpreter','latex')
ylabel('$\rho$','Interpreter','latex')
box on
legend('\sigma = 10^{-3}','\sigma = 10^{-2}','\sigma = 0.05','\sigma = 0.1')
title(['$\eta = ',num2str(eta),'$'],'Interpreter','latex')


%% Profile of the lateral matrix
ths = -pi:0.02:pi;
Mphi = @(x) (sin(abs(x)) + (pi - abs(x)).*cos(x))/4/pi;

figure
plot(ths,Mphi(ths))
xlim([-pi,pi])
xlabel("$\phi - \phi'$",'Interpreter','latex')
ylabel("$M(\phi - \phi')$",'Interpreter','latex')
set(gca,'XTick',(-1:0.5:1)*pi,'XTickLabel',{'\pi','-\pi/2','0','\pi/2','\pi'})