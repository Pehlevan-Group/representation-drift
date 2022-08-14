% This program plot an illustration of 1D slice through 2D lattice
% 

close all
clear


param.Nlbd = 5;       % number of different scales/spacing,default 5
param.Nthe = 6;         % number of rotations
param.Nx =  3;          % offset of x-direction
param.Ny = 3;           % offset of y-direction
param.Ng = param.Nlbd*param.Nthe*param.Nx*param.Ny;   % total number of grid cells

% example 2D grid fields
lbd = 0.25;
theta = 1/6*pi;
r0 = [0;0];
ps = 100;
gv = PlaceCellhelper.gridModule(lbd,theta,r0,ps);

gM = reshape(gv,[ps,ps]);
% figure
% imagesc(gM)

% all the slices
ori = 1/12*pi;
gridSlice1 = PlaceCellhelper.gridModuleSlices(lbd,theta,r0,200,ori);


lbd2 = 0.25*1.42;
r0 = [5;5]/5*lbd2;
% theta = pi/12; 
gridSlice2 = PlaceCellhelper.gridModuleSlices(lbd2,theta,r0,200,ori);


% figure
% plot(gridSlice2)


%% plot and save figure
sFolder = '../figures';

blues = brewermap(11,'Blues');
reds = brewermap(11,'Reds');


% example grid fields
grid_fig = figure;
set(grid_fig,'color','w','Units','inches')
pos(3)=3.2;  %17.4; %8.5;%pos(3)*1.5;
pos(4)=2.8;%pos(4)*1.5;
set(grid_fig,'Position',pos)

imagesc(gM)
xlabel('X position','FontSize',16)
ylabel('Y position','FontSize',16)
set(gca,'LineWidth',1,'FontSize',16,'XTickLabel','','YTickLabel','')

% figPre = '1DgridCell_slice';
% prefix = [figPre, 'heatmap'];
% saveas(grid_fig,[sFolder,filesep,prefix,'.fig'])
% print('-depsc',[sFolder,filesep,prefix,'.eps'])


% ****************************************
% two example slices
% ****************************************
slice_fig = figure;
set(slice_fig,'color','w','Units','inches')
pos(3)=3.2;  %17.4; %8.5;%pos(3)*1.5;
pos(4)=2.8;%pos(4)*1.5;
set(slice_fig,'Position',pos)
hold on
plot(gridSlice1,'LineWidth',1.5,'Color',blues(9,:))
plot(gridSlice2,'LineWidth',1.5,'Color',reds(9,:))
box on
lg = legend('grid cell 1', 'grid cell 2');
set(lg,'FontSize',16)
xlabel('Position','FontSize',20)
ylabel('Response','FontSize',20)
set(gca,'LineWidth',1,'FontSize',16,'XTickLabel','','YTickLabel','')

% figPre = '1DgridCell_slice';
% prefix = [figPre, 'example'];
% saveas(slice_fig,[sFolder,filesep,prefix,'.fig'])
% print('-depsc',[sFolder,filesep,prefix,'.eps'])
