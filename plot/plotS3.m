% Figure S3
% plot the correlation coefficient of PV for three different noise
% scenarios


%%  Fig S3A simple 1d place cell model

% load the data
dFile = '../data_in_paper/1D_place_different_noise.mat';
load(dFile)

blues = brewermap(11,'Blues');
reds = brewermap(11,'Reds');
greys = brewermap(11,'Greys');
fig_colors = [blues([7,11],:);reds([7,11],:);greys([7,11],:)];


time_points = round(total_iter/param.step);

% calculate the autocorrelation coefficients population vectors
% pvCorr = zeros(size(Yt,3),size(Yt,2)); 
% [~,neuroInx] = sort(peakInx(:,inxSel(1)));
    
f_pvCorr = figure;
set(f_pvCorr,'color','w','Units','inches')
pos(3)=3.5;  
pos(4)=2.8;
set(f_pvCorr,'Position',pos)

for phase = 1:3
    pvCorr = zeros(size(all_Yts{phase},3),size(all_Yts{phase},2)); 
    for i = 1:size(all_Yts{phase},3)
        for j = 1:size(all_Yts{phase},2)
            temp = all_Yts{phase}(:,j,i);
            C = corrcoef(temp,all_Yts{phase}(:,j,1));
            pvCorr(i,j) = C(1,2);
        end
    end
    PV_corr_coefs{phase} = pvCorr;
    % plot
    fh = shadedErrorBar((1:size(pvCorr,1))'*param.step,pvCorr',{@mean,@std});
    box on
    set(fh.edge,'Visible','off')
    fh.mainLine.LineWidth = 3;
    fh.mainLine.Color = fig_colors(2*phase-1,:);
    fh.patch.FaceColor = fig_colors(2*phase,:);
end
lg = legend('Full model','Forward noise', 'recurrent noise');
set(lg, 'FontSize', 12)
xlim([0,200]*param.step)
xlabel('$t$','Interpreter', 'latex','FontSize',16)
ylabel('PV correlation','FontSize',16)
set(gca,'FontSize',16,'LineWidth',1)


%% Fig S3A simple 1d place cell model
% Nois is introduced to either feedforward, recurrent or all matrices

% The figure can be generated directly by running
% 'ring_model_three_phases.m' or by loading the data generated
dFile = '../data_in_paper/ring_model_different_noise.mat';
load(dFile,'total_iter','params','all_Yts')

blues = brewermap(11,'Blues');
reds = brewermap(11,'Reds');
greys = brewermap(11,'Greys');
fig_colors = [blues([7,11],:);reds([7,11],:);greys([7,11],:)];

time_points = round(total_iter/params.record_step);
   
f_pvCorr = figure;
set(f_pvCorr,'color','w','Units','inches')
pos(3)=3.5;  
pos(4)=2.8;
set(f_pvCorr,'Position',pos)

for phase = 1:3
    pvCorr = zeros(size(all_Yts{phase},3),size(all_Yts{phase},2)); 
    for i = 1:size(all_Yts{phase},3)
        for j = 1:size(all_Yts{phase},2)
            temp = all_Yts{phase}(:,j,i);
            C = corrcoef(temp,all_Yts{phase}(:,j,1));
            pvCorr(i,j) = C(1,2);
        end
    end
    PV_corr_coefs{phase} = pvCorr;
    % plot
    fh = shadedErrorBar((1:size(pvCorr,1))'*params.record_step,pvCorr',{@mean,@std});
    box on
    set(fh.edge,'Visible','off')
    fh.mainLine.LineWidth = 3;
    fh.mainLine.Color = fig_colors(2*phase-1,:);
    fh.patch.FaceColor = fig_colors(2*phase,:);
end
lg = legend('Full model','Forward noise', 'Recurrent noise');
set(lg,'FontSize',10)

% ylim([0.25,1])
xlim([0,100]*params.record_step)
xlabel('Time','FontSize',16)
ylabel('PV correlation','FontSize',16)
set(gca,'FontSize',16,'LineWidth',1)