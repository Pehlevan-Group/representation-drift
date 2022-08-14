
%
psi = 0.01:0.01:pi/2;
mu2 = sqrt((2*psi - sin(2*psi))./(4*psi + 2*psi.*cos(2*psi) - 3*sin(2*psi)));
mu_hat = mu2.^(2*psi - sin(psi))/4/pi;
amp = mu2.*(1-cos(psi));
alp2 = cos(psi).*(2*psi - sin(2*psi))./4./(sin(psi) - psi.*cos(psi));

% add diffusion constant



%% plot the figures

% graphics setting
figWidth = 3.2;
figHeight = 2.8;
lineWd = 2;
symbSize = 4;
labelSize = 20;
axisSize = 16;
axisWd = 1;

figPre = 'ring_theory_';
sFolder = './figures';

% figure size, weight and height
pos(3)=figWidth;  
pos(4)=figHeight;

f_psi_amp= figure;
set(f_psi_amp,'color','w','Units','inches','Position',pos)
plot(psi,amp,'LineWidth',lineWd);
xlim([0 pi/2])
ax = gca;
ax.XTick =[0 pi/4 pi/2];
ax.XTickLabel = {'0', '\pi/4', '\pi/2'};

xlabel('$\psi$','Interpreter','latex','FontSize',labelSize)
ylabel('$\mu(1-\cos\psi)$','Interpreter','latex','FontSize',labelSize)
set(gca,'FontSize',axisSize)

prefix = [figPre, 'psi_amp'];
saveas(f_psi_amp,[sFolder,filesep,prefix,'.fig'])
print('-depsc',[sFolder,filesep,prefix,'.eps'])


f_psi_alp = figure;
set(f_psi_alp,'color','w','Units','inches','Position',pos)

plot(psi,alp2,'LineWidth',lineWd)
xlim([0 pi/2])
ax = gca;
ax.XTick =[0 pi/4 pi/2];
ax.XTickLabel = {'0', '\pi/4', '\pi/2'};
xlabel('$\psi$','Interpreter','latex','FontSize',labelSize)
ylabel('$\alpha^2$','Interpreter','latex','FontSize',labelSize)
set(gca,'FontSize',axisSize)
prefix = [figPre, 'psi_alp'];
saveas(f_psi_alp,[sFolder,filesep,prefix,'.fig'])
print('-depsc',[sFolder,filesep,prefix,'.eps'])



alp_amp = figure;
set(alp_amp,'color','w','Units','inches','Position',pos)

plot(alp2,amp,'LineWidth',lineWd)
xlabel('$\alpha^2$','Interpreter','latex','FontSize',labelSize)
ylabel('$\mu(1-\cos\psi)$','Interpreter','latex','FontSize',labelSize)
set(gca,'FontSize',axisSize)
prefix = [figPre, 'alp_amp'];
saveas(alp_amp,[sFolder,filesep,prefix,'.fig'])
print('-depsc',[sFolder,filesep,prefix,'.eps'])




alp_muhat = figure;
set(alp_muhat,'color','w','Units','inches','Position',pos)

plot(alp2,mu_hat,'LineWidth',lineWd)
xlabel('$\alpha^2$','Interpreter','latex','FontSize',labelSize)
ylabel('$\mu(2\psi - \sin(2\psi))$','Interpreter','latex','FontSize',labelSize)
set(gca,'FontSize',axisSize)
prefix = [figPre, 'alp_muhat'];
saveas(alp_muhat,[sFolder,filesep,prefix,'.fig'])
print('-depsc',[sFolder,filesep,prefix,'.eps'])



f_alp_muh = figure;
set(f_alp_muh,'color','w','Units','inches','Position',pos)

plot(amp,mu_hat,'LineWidth',lineWd)
xlabel('$\mu(1-\cos\psi)$','Interpreter','latex','FontSize',labelSize)
ylabel('$\mu(2\psi - \sin(2\psi))$','Interpreter','latex','FontSize',labelSize)
set(gca,'FontSize',axisSize)

prefix = [figPre, 'amp_muhat'];
saveas(f_alp_muh,[sFolder,filesep,prefix,'.fig'])
print('-depsc',[sFolder,filesep,prefix,'.eps'])




%% Gamma
gammas = pi/6*(36*psi + 24*psi.*cos(2*psi) - 28*sin(2*psi) - sin(4*psi))./(2*psi-sin(2*psi)).^2;

fig_gm = figure;
set(fig_gm,'color','w','Units','inches','Position',pos)

plot(psi,gammas,'LineWidth',lineWd)
xlim([0 pi/2])
ax = gca;
ax.XTick =[0 pi/4 pi/2];
ax.XTickLabel = {'0', '\pi/4', '\pi/2'};
xlabel('$\psi$','Interpreter','latex','FontSize',labelSize)
ylabel('$\gamma$','Interpreter','latex','FontSize',labelSize)
set(gca,'FontSize',axisSize)
prefix = [figPre, 'psi_gamma'];
saveas(fig_gm,[sFolder,filesep,prefix,'.fig'])
print('-depsc',[sFolder,filesep,prefix,'.eps'])



%% Theory of Diffusion constants

% how Ds depend on alpha
sig = 0.01;
etas = [1e-3,1e-2,1e-1];
alp = 0.0:0.01:0.99;

eta_Ds = figure;
set(eta_Ds,'color','w','Units','inches','Position',pos)

hold on
for i = 1:length(etas)
    Ds = gammas.*etas(i)^2 + etas(i).*sig.^2./mu_hat.^2;
    plot(alp2,Ds,'Color',blues(1+3*i,:),'LineWidth',lineWd)
end
box on
hold off

ylim([1e-6,1e-1])
legend('\eta = 10^{-3}','\eta = 10^{-2}','\eta = 10^{-1}')
xlabel('$\alpha^2$','Interpreter','latex','FontSize',labelSize)
ylabel('$D$','Interpreter','latex','FontSize',labelSize)
set(gca,'FontSize',axisSize,'YTick',10.^(-5:2:-1),'YScale','log')

prefix = [figPre, 'sig_ds'];
saveas(eta_Ds,[sFolder,filesep,prefix,'.fig'])
print('-depsc',[sFolder,filesep,prefix,'.eps'])

%% % how Ds depend on sigma
blues = brewermap(11,'Blues');
sigs = [1e-3,1e-2,1e-1];
eta = 0.01;
alp = 0.0:0.01:0.99;

sig_Ds = figure;
set(sig_Ds,'color','w','Units','inches','Position',pos)

hold on
for i = 1:length(sigs)
    Ds = gammas.*eta.^2 + eta.*sigs(i).^2./mu_hat.^2;
    plot(alp2,Ds,'Color',blues(1+3*i,:),'LineWidth',lineWd)
end
hold off
box on
ylim([1e-6,1e-1])
legend('\sigma = 10^{-3}','\sigma = 10^{-2}','\sigma = 10^{-1}')
xlabel('$\alpha^2$','Interpreter','latex','FontSize',labelSize)
ylabel('$D$','Interpreter','latex','FontSize',labelSize)
set(gca,'FontSize',axisSize,'YTick',10.^(-5:2:-1),'YScale','log')

prefix = [figPre, 'sig_ds'];
saveas(sig_Ds,[sFolder,filesep,prefix,'.fig'])
print('-depsc',[sFolder,filesep,prefix,'.eps'])


%% How Ds depend on the relative amplitude of Rf
eta = 0.01;
sig = 0.01;
blues = brewermap(11,'Blues');

Ds = gammas.*eta.^2 + eta.*sig.^2./mu_hat;

d_ampl = figure;
set(d_ampl,'color','w','Units','inches','Position',pos)
plot(amp,Ds,'Color',blues(9,:),'LineWidth',lineWd)
legend(['\eta = ',num2str(eta),', \sigma=',num2str(sig)])
xlabel('$\mu(1 - \cos\psi$','Interpreter','latex','FontSize',labelSize)
ylabel('$D$','Interpreter','latex','FontSize',labelSize)
set(gca,'FontSize',axisSize,'YScale','log')
