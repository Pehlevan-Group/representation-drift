% this script defines the default graphics settings
% re = [];  %red
% gr = [];  %green
% bl = [];  %blue
% cy = [];  %cyan
% or = [];  %orange
% gy = [];  %grey
set(groot,{'DefaultAxesXColor','DefaultAxesYColor','DefaultAxesZColor'},{'k','k','k'})
set(groot, 'defaultLineLineWidth',2);
set(groot, 'DefaultAxesColor', 'remove');
% set(groot, 'DefaultTextInterpreter', 'lateX');
% set(groot, 'DefaultAxesTickLabelInterpreter', 'lateX');
% set(groot, 'DefaultAxesFontName', 'Helvitica');
% set(groot, 'DefaultLegendInterpreter', 'lateX');
set(groot, 'DefaultAxesLineWidth', 1);
set(groot, 'DefaultFigureInvertHardcopy', 'on');
set(groot, 'DefaultAxesLabelFontSizeMultiplier', 1.2);

set(groot, 'DefaultAxesFontSize', 16);
set(groot, 'DefaultLegendBox', 'off');
set(groot, 'DefaultLegendFontSize',20);

set(groot,'DefaultFigurePaperUnits', 'inches', ...
           'DefaultFigureUnits', 'inches', ...
           'DefaultFigurePaperPosition', [0, 0, 6.1, 4.6], ...
           'DefaultFigurePaperSize', [6.1, 4.6], ...
           'DefaultFigurePosition', [0.1, 0.1, 6, 4.5]);

% the following should be uncommented 
% set(groot, 'DefaultAxesFontSize', 24);
% set(groot, 'DefaultLegendBox', 'off');
% set(groot, 'DefaultLegendFontSize',20);
% % 
% set(groot,'DefaultFigurePaperUnits', 'inches', ...
%            'DefaultFigureUnits', 'inches', ...
%            'DefaultFigurePaperPosition', [0, 0, 6.1, 4.6], ...
%            'DefaultFigurePaperSize', [6.1, 4.6], ...
%            'DefaultFigurePosition', [0.1, 0.1, 6, 4.5]);