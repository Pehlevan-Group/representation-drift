function [Yt, param] = grid_cell_stochastic_update_snsm(X,total_iter, param)

ystart = zeros(param.num_grid,param.batch_size);  % inital of the output
time_points = round(total_iter/param.step);
Yt = nan(param.num_grid,param.ps,time_points);

% generate position code by the grid cells    
for i = 1:total_iter
    positions = X(:,randperm(param.ps,param.batch_size));
    states = PlaceCellhelper.nsmDynBatch(positions,ystart, param);
    y = states.Y;

    % noisy update weight matrix
    param.W = (1-param.learn_rate)*param.W + param.learn_rate*y*positions'/param.BatchSize + ...
        sqrt(param.learn_rate)*param.noiseW.*randn(param.num_grid,param.num_fields); 
    param.M = max(0,(1-param.learn_rate)*param.M + param.learn_rate*y*y'/param.BatchSize + ...
        sqrt(param.learn_rate)*param.noiseM.*randn(param.num_grid,param.num_grid));
    param.b = (1-param.learn_rate)*param.b + param.learn_rate*sqrt(param.alpha)*mean(y,2);

    % store and check representations
    if mod(i, param.step) == 0
        Y0 = zeros(param.num_grid,size(X,2));
        states_fixed = PlaceCellhelper.nsmDynBatch(X,Y0, param);
        Yt(:,:,round(i/param.step)) = states_fixed.Y;
    end
end

end
