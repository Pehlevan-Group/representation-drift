% run the decoding performance vs decoding learning rate

lr = 0.02;
sig = 0.01;
Np = 100;
alp = 0;

allDecoRate = 10.^(-1:0.2:2)*lr;

decodeErr = nan(length(allDecoRate),2);

for i = 1:length(allDecoRate)
    readError = ringDecoding(allDecoRate(i), Np, alp, sig, lr);
    decodeErr(i,:) = readError;
end


%%  plot the figure
blues = brewermap(11,'Blues');
greys = brewermap(11,'Greys');

pos(3)=3.2;  
pos(4)=2.8;

fig_err_rate= figure;
set(fig_err_rate,'color','w','Units','inches','Position',pos)

errorbar(allDecoRate/lr,decodeErr(:,1),decodeErr(:,2),'o-','MarkerSize',6,'MarkerFaceColor',...
    greys(9,:),'MarkerEdgeColor',greys(9,:),'Color',greys(9,:),'LineWidth',...
    1.5,'CapSize',0)
box on
% xlim([0.1,3])
xlabel('$\eta_{out}/\eta $','Interpreter','latex','FontSize',20)
ylabel('$\epsilon$','Interpreter','latex','FontSize',20)
set(gca,'FontSize',16,'LineWidth',1,'XScale','log','YScale','linear',...
    'XTick',10.^(-1:1:2))