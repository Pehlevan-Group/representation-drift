function gridFields = slice2Dgrid(ps,Nlbd,Nthe,Nx,Ny,ori,vargin)

param.ps =  ps;         % number of positions along each dimension
param.Nlbd = Nlbd;      % number of different scales/spacing
param.Nthe = Nthe;      % number of rotations
param.Nx =  Nx;         % offset of x-direction
param.Ny = Ny;          % offset of y-direction
param.Ng = param.Nlbd*param.Nthe*param.Nx*param.Ny;   % total number of grid cells

param.baseLbd = 0.2;    % spacing of smallest grid RF, default 0.28
param.sf =  1.42;       % scaling factor between adjacent module

param.ori = ori;        % slicing orientation

if nargin > 6
    type = vargin{1};  % for remapping
else
    type = 'regular';  % regular slicing
end
    

param.lbds = param.baseLbd*(param.sf.^(0:param.Nlbd-1));    % all the spacings of different modules
param.thetas =(0:param.Nthe-1)*pi/3/param.Nthe;             % random sample rotations
param.x0  = (0:param.Nx-1)'/param.Nx*param.lbds;            % offset of x
param.y0  = (0:param.Ny-1)'/param.Ny*param.lbds;            % offset of y

if strcmp(type,'shifted')
    % using shifted 2D grid fields for slicing
    % each module has a random shift, within module the shift is the same
    gridFields = nan(param.ps,param.Ng);
    count = 1;    % concantenate the grid cells
    for i = 1:param.Nlbd
        % random shift
        dr = rand*0.4 + 0.1;
        dth = rand*2*pi;
        for j = 1: param.Nthe
            for k = 1:param.Nx
                for l = 1:param.Ny
%                     r = [i/param.ps;j/param.ps];
                    r0 = [param.x0(k,i);param.y0(l,i)];
%                     r0 = rand(2,1)*param.lbds(i); % modified 7/7/21
                    gridFields(:,count) = PlaceCellhelper.gridModuleSlicesShifted(param.lbds(i),...
                        param.thetas(j),r0,param.ps,param.ori,dr, dth);
                    count = count +1;
                end
            end
        end
    end
else
    % generate a Gramian of the grid fields
    gridFields = nan(param.ps,param.Ng);
    count = 1;    % concantenate the grid cells
    for i = 1:param.Nlbd
        for j = 1: param.Nthe
            for k = 1:param.Nx
                for l = 1:param.Ny
%                     r = [i/param.ps;j/param.ps];
                    r0 = [param.x0(k,i);param.y0(l,i)];
%                     r0 = rand(2,1)*param.lbds(i); % modified 7/7/21
                    gridFields(:,count) = PlaceCellhelper.gridModuleSlices(param.lbds(i),...
                        param.thetas(j),r0,param.ps,param.ori);
                    count = count +1;
                end
            end
        end
    end
end

end