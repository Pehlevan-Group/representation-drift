% This program simulate how the learning rate, noise level and eigen
% spectrum change the relative diffusion constants

saveFolder = './figures';
type = 'psp'; % or 'psp'
tau = 0.5;
learnType = 'offline';

% plot setting
defaultGraphicsSetttings
blues = brewermap(11,'Blues');
rb = brewermap(11,'RdBu');
set1 = brewermap(9,'Set1');

%% learning rate depedent, keep eigen spectrum and noise level, tau the same
dimIn = 10;
dimOut = 3;

eigs = [4.5,3,1.5,ones(1,7)*0.02];
% eigs = [5,3,1.3,ones(1,7)*0.05];
% tau = 0.5;
% eigs = [2 1 0.1];
noiseStd = 0.1; % 0.005 for expansion, 0.1 for psp

learnRate = 1*10.^(-1:0.1:-1);
diffEu = nan(length(learnRate),2);  %store the mean and standard deviation of the diffusion constants
diffThe = nan(length(learnRate),2);
alpha_eu = nan(length(learnRate),2); % store the average and std of exponent
alpha_the = nan(length(learnRate),2);  
for i = 1:length(learnRate)
    [deu,dth] = diffConstants(dimIn, dimOut, eigs,noiseStd,learnRate(i),tau,learnType);
    diffEu(i,:) = [mean(deu(:,1)),std(deu(:,1))];
    alpha_eu(i,:) = [mean(deu(:,2)),std(deu(:,2))];
    diffThe(i,:) = [mean(dth(:,1)),std(dth(:,1))];
    alpha_the(i,:) = [mean(dth(:,2)),std(dth(:,2))];
end

alp_dr = [ones(length(learnRate),1),log10(learnRate')]\log10(diffEu(:,1));
alp_dthe =[ones(length(learnRate),1), log10(learnRate')]\log10(diffThe(:,1));

%plot
X0 = [ones(length(learnRate),1),log10(learnRate')];
figure
eh1 = errorbar(learnRate',diffEu(:,1),diffEu(:,2),'o','MarkerSize',8,'MarkerFaceColor',...
    blues(10,:),'Color',blues(10,:),'LineWidth',2,'CapSize',0);
hold on
lh1 = plot(learnRate,10.^(X0*alp_dr),'LineWidth',3,'Color',blues(10,:));
eh2 = errorbar(learnRate',diffThe(:,1),diffThe(:,2),'o','MarkerSize',8,'MarkerFaceColor',...
    rb(2,:),'Color',rb(2,:),'LineWidth',2,'CapSize',0);
lh2 = plot(learnRate,10.^(X0*alp_dthe),'LineWidth',3,'Color',rb(2,:));
hold off
% plot(learnRate',diffEu(:,1),'LineWidth',4)
% hold on
% plot(learnRate',diffThe(:,1),'LineWidth',4)
% hold off
lg = legend([eh1,eh2],['$D_r, \gamma_r = ',num2str(round(alp_dr(2)*100)/100),'$'],['$D_{\gamma},\gamma{\theta} =',...
    num2str(round(alp_dthe(2)*100)/100),'$'],'Location','northwest');
% lg = legend('$D_r$','$D_{\theta}$','Location','northwest');
set(lg,'Interpreter','Latex')
legend boxoff
xlabel('learning rate $(\eta)$','Interpreter','latex','FontSize',28)
ylabel('diffusion constant','FontSize',28)
set(gca,'LineWidth',1.5,'FontSize',24,'XScale','log','YScale','log')


prefix = [learnType,'_',type,'_diff_const_lr_noise',num2str(noiseStd),'_tau',num2str(tau),date];
saveas(gcf,[saveFolder,filesep,prefix,'.fig'])
print('-depsc',[saveFolder,filesep,prefix,'.eps'])


% change of exponents
figure
eh1 = errorbar(learnRate',alpha_eu(:,1),alpha_eu(:,2),'o','MarkerSize',8,'MarkerFaceColor',...
    blues(10,:),'Color',blues(10,:),'LineWidth',2,'CapSize',0);
hold on
% lh1 = plot(learnRate,10.^(X0*alp_dr),'LineWidth',3,'Color',blues(10,:));
eh2 = errorbar(learnRate',alpha_the(:,1),alpha_the(:,2),'o','MarkerSize',8,'MarkerFaceColor',...
    rb(2,:),'Color',rb(2,:),'LineWidth',2,'CapSize',0);
% lh2 = plot(learnRate,10.^(X0*alp_dthe),'LineWidth',3,'Color',rb(2,:));
hold off
xlabel('learning rate $(\eta)$','Interpreter','latex','FontSize',28)
ylabel('$\alpha$','Interpreter','latex','FontSize',28)
set(gca,'LineWidth',1.5,'FontSize',24,'XScale','log')
prefix = [learnType,'_',type,'_exponent_lr_noise',num2str(noiseStd),'_tau',num2str(tau),date];
saveas(gcf,[saveFolder,filesep,prefix,'.fig'])
print('-depsc',[saveFolder,filesep,prefix,'.eps'])


dataName = [type,'_',learnType,'_std',num2str(noiseStd),'_',date];
save(fullfile('./data',dataName))

%% noise dependence

dimIn = 10;
dimOut = 3;
noiseStd = 5*10.^(-2:0.05:-1.6);
% noiseStd = 5*10.^(-5:0.1:-3);
eigs = [5,3,1.3,ones(1,7)*0.1];
% eigs = [2 1 0.1];
% noiseStd = 0.005;
% tau = 0.1;

learnRate = 0.05;  %
diffEu_noise = nan(length(noiseStd),2);  %store the mean and standard deviation of the diffusion constants
diffThe_noise = nan(length(noiseStd),2);
alpha_eu_noise = nan(length(learnRate),2); % store the average and std of exponent
alpha_the_noise = nan(length(learnRate),2);  
for i = 1:length(noiseStd)
    [deu,dth,perturb] = diffConstants(dimIn, dimOut, eigs,noiseStd(i),learnRate,tau,learnType);
    diffEu_noise(i,:) = [mean(deu(:,1)),std(deu(:,1))];
    alpha_eu_noise(i,:) = [mean(deu(:,2)),std(deu(:,2))];
    diffThe_noise(i,:) = [mean(dth(:,1)),std(dth(:,1))];
    alpha_the_noise(i,:) = [mean(dth(:,2)),std(dth(:,2))];
end

% linear regression to estimate the power law exponent
alp_dr = [ones(length(noiseStd),1),log10(noiseStd')]\log10(diffEu_noise(:,1));
alp_dthe =[ones(length(noiseStd),1), log10(noiseStd')]\log10(diffThe_noise(:,1));

%plot
X0 = [ones(length(noiseStd),1),log10(noiseStd')];
figure
eh1 = errorbar(noiseStd',diffEu_noise(:,1),diffEu_noise(:,2),'o','MarkerSize',8,'MarkerFaceColor',...
    blues(10,:),'Color',blues(10,:),'LineWidth',2,'CapSize',0);
hold on
lh1 = plot(noiseStd,10.^(X0*alp_dr),'LineWidth',3,'Color',blues(10,:));
eh2 = errorbar(noiseStd',diffThe_noise(:,1),diffThe_noise(:,2),'o','MarkerSize',8,'MarkerFaceColor',...
    rb(2,:),'Color',rb(2,:),'LineWidth',2,'CapSize',0);
lh2 = plot(noiseStd,10.^(X0*alp_dthe),'LineWidth',3,'Color',rb(2,:));
hold off
lg = legend([eh1,eh2],['$D_r, \gamma_r = ',num2str(round(alp_dr(2)*100)/100),'$'],['$D_{\theta},\gamma_{\theta} =',...
    num2str(round(alp_dthe(2)*100)/100),'$'],'Location','northwest');
set(lg,'Interpreter','Latex')
legend boxoff
xlabel('noise std $(\sigma)$','Interpreter','latex','FontSize',28)
ylabel('diffusion constant','FontSize',28)
set(gca,'LineWidth',1.5,'FontSize',24,'XScale','log','YScale','log')

prefix = [learnType,'_',type,'_diff_const_noise_lr',num2str(learnRate),'_tau',num2str(tau),date];
saveas(gcf,[saveFolder,filesep,prefix,'.fig'])
print('-depsc',[saveFolder,filesep,prefix,'.eps'])


% change of exponents
figure
eh1 = errorbar(noiseStd',alpha_eu_noise(:,1),alpha_eu_noise(:,2),'o','MarkerSize',8,'MarkerFaceColor',...
    blues(10,:),'Color',blues(10,:),'LineWidth',2,'CapSize',0);
hold on
% lh1 = plot(learnRate,10.^(X0*alp_dr),'LineWidth',3,'Color',blues(10,:));
eh2 = errorbar(noiseStd',alpha_the_noise(:,1),alpha_the_noise(:,2),'o','MarkerSize',8,'MarkerFaceColor',...
    rb(2,:),'Color',rb(2,:),'LineWidth',2,'CapSize',0);
% lh2 = plot(learnRate,10.^(X0*alp_dthe),'LineWidth',3,'Color',rb(2,:));
hold off
ylim([0,1])
xlabel('learning rate $(\sigma)$','Interpreter','latex','FontSize',28)
ylabel('$\alpha$','Interpreter','latex','FontSize',28)
set(gca,'LineWidth',1.5,'FontSize',24,'XScale','log')
prefix = [learnType,'_',type,'_exponent_lr',num2str(learnRate),'_tau',num2str(tau),date];
saveas(gcf,[saveFolder,filesep,prefix,'.fig'])
print('-depsc',[saveFolder,filesep,prefix,'.eps'])

% save all the variables
dataName = [type,'_',learnType,'_lr',num2str(learnRate),'_',date];
save(fullfile('./data',dataName))

%% scaling factor of learning rate
dimIn = 3;
dimOut = 10;
noiseStd = 0.005;
eigs = [2 1 0.1];

learnRate = 0.02;
taus = 0.05:0.05:0.5;
diffEu_tau = nan(length(taus),2);  %store the mean and standard deviation of the diffusion constants
diffThe_tau = nan(length(taus),2);
for i = 1:length(taus)
    [deu,dth] = diffConstants(dimIn, dimOut, eigs,noiseStd,learnRate, taus(i));
    diffEu_tau(i,:) = [mean(deu),std(deu)];
    diffThe_tau(i,:) = [mean(dth),std(dth)];
end

%plot
figure
plot(taus',diffEu_tau(:,1),'LineWidth',4)
hold on
plot(taus',diffThe_tau(:,1),'LineWidth',4)
hold off
lg = legend('$D_r$','$D_{\theta}$','Location','northwest');
set(lg,'Interpreter','Latex')
legend boxoff
xlabel('scaling factor $(\tau)$','Interpreter','latex','FontSize',28)
ylabel('diffusion constant','FontSize',28)
set(gca,'LineWidth',1.5,'FontSize',24,'XScale','linear','YScale','log')

prefix = [type,'_diff_tau_lr',num2str(learnRate),'noise',num2str(noiseStd)];
saveas(gcf,[saveFolder,filesep,prefix,'.fig'])
print('-depsc',[saveFolder,filesep,prefix,'.eps'])
