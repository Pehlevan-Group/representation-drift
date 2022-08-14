function gridFields_background = background_grid_1d(param)
% this function return a big matrix containing different grid maps
rotations = rand(100,1)*pi;
gridFields_background = nan(100*param.ps,param.Ng);
for i = 1:100
    gridFields = slice2Dgrid(param.ps,param.Nlbd,param.Nthe,param.Nx,param.Ny,rotations(i));
    gridFields_background(1+(i-1)*param.ps: i*param.ps,:) = gridFields;
end

end