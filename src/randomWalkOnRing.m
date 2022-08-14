% random walk on a ring


tot_iter = 2e4;
% D = 1e-2;
D = std(dphi(1,:));  % from simulation
Np = 3;

moveDirect = sign(randn(tot_iter,Np));
xs = cumsum(moveDirect,1)*sqrt(D);
xs2 = cumsum(randn(tot_iter,Np)*D) + rand(1,3)*2*pi;

figure
plot(xs,'LineWidth',1.5)

figure
plot(xs2,'LineWidth',1.5)
set(gcf,'color','w')
xlabel('time')
ylabel('$\phi$','Interpreter','latex')
set(gca,'FontSize',20,'LineWidth',1.5)


% mean squre displacement
modelPk = mod(xs2,2*pi);   % 11/21/2020

msdsModel = nan(floor(tot_iter/2),Np);
for i = 1:floor(tot_iter/2)
    diffLag = min(abs(modelPk(i+1:end,:) - modelPk(1:end-i,:)),...
        2*pi - abs(modelPk(i+1:end,:) - modelPk(1:end-i,:)) );
    msdsModel(i,:) = nanmean(diffLag.*diffLag,1);
end

DsModel = PlaceCellhelper.fitLinearDiffusion(msdsModel,step,'linear');

figure
plot(msdsModel)
xlabel('time')
ylabel('$\langle\Delta\phi^2\rangle$','Interpreter','latex')
set(gca,'FontSize',20,'LineWidth',1.5)


figure
histogram(randn(tot_iter,1)*D,'Normalization','pdf')
xlabel('$\Delta\phi$','Interpreter','latex')
ylabel('pdf')


% correlation coefficient
figure
hold on
for i = 1:k
    acf = autocorr(xs2(:,i),'NumLags',2000);
    plot(acf)
end
hold off
box on
xlabel('$\Delta t$','Interpreter','latex')
ylabel('auto corr. coef.')
set(gca,'FontSize',20,'LineWidth',1.5)


figure
plot(xs2(1:1000,2),xs2(1001:2000,2),'o')

corr(xs2(1:1000,1),xs2(1001:2000,1))
%% Analytical solution of the drift
eta = 0.01;
sig = 0.002;
phi = pi/3;
thetas = rand(tot_iter,1)*2*pi;
ys = max(cos(thetas-phi),0).*sin(thetas-phi)*eta/4 +  sqrt(eta)*sig*(randn(tot_iter,1).*cos(phi) ...
    + randn(tot_iter,1).*sin(phi));
% ys = max(cos(thetas),0).*sin(thetas)*eta/4 + sqrt(eta)*sig*randn(tot_iter,1);


figure
histogram(ys,'Normalization','pdf')
xlabel('$\Delta \phi$','interpreter','latex')
ylabel('pdf','interpreter','latex')
title (['$\eta = ',num2str(eta),',\sigma = ',num2str(sig),'$'],'Interpreter','latex' )