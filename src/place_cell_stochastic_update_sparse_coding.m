function [output, param] = place_cell_stochastic_update_sparse_coding(X,total_iter, param, dim)
% runing the population codes, centroid position and noisy weight update and return 
% the learned representations, using sparse coding method

if exist('dim','var')
    DIM = dim;  % 1D or 2 D
else
    DIM = 1;
end

time_points = round(total_iter/param.step);
if DIM == 1
    L = param.ps;
    pkCenterMass = nan(param.Np,time_points);
elseif DIM ==2
    L = param.ps^2;
    pkCenterMass = nan(param.Np,2, time_points);

end

ystart = zeros(param.Np,param.BatchSize);  % inital of the output
Yt = nan(param.Np,L,time_points);
pkMas = nan(param.Np,time_points);  
pkAmp = nan(param.Np,time_points); 
pks = nan(param.Np,time_points);

% generate position code by the grid cells    
for i = 1:total_iter
    positions = X(:,randperm(L,param.BatchSize));
    states = PlaceCellhelper.sparseCodingDynBatch(positions,ystart, param);
    y = states.Y;
    
    % try input noise
    y_noise = y + randn(size(y))*param.input_std;

    % noisy update weight matrix
%     param.W = param.W + param.learnRate*(y*positions'- y*y'*param.W)/param.BatchSize ...
%         +sqrt(param.learnRate)*param.noiseW.*randn(param.Np,param.Ng);
    param.W = param.W + param.learnRate*(y*positions'- y*y'*param.W)/param.BatchSize;
    param.W = max(param.W,0);      % non-negative
    param.W = normalize_matrix(param.W,1); % normalize each row to be length 1
    % introduce noise after normalization
%     param.W = param.W + sqrt(param.learnRate)*param.noiseW.*randn(param.Np,param.Ng);
    
    % suggested by Cengiz
    param.M = (1-param.learnRate)*param.M + param.learnRate*(param.W*param.W' - eye(param.Np))...
        + sqrt(param.learnRate)*param.noiseM.*randn(param.Np);
    
    % store and check representations
    if mod(i, param.step) == 0
        Y0 = zeros(param.Np,size(X,2));
        states_fixed = PlaceCellhelper.sparseCodingDynBatch(X,Y0, param);
        flag = sum(states_fixed.Y > param.ampThd,2) > 3;   % only those neurons that have multiple non-zeros considered
        [~,pkInx] = sort(states_fixed.Y,2, 'descend');
        temp = states_fixed.Y((pkInx(:,1)-1)*param.Np + (1:param.Np)');
        pkAmp(flag,round(i/param.step)) = temp(flag);
        pks(flag,round(i/param.step)) = pkInx(flag,1);
        Yt(:,:,round(i/param.step)) = states_fixed.Y;

        % store the center of mass
        if DIM == 1
            [pkCM, aveMass] = PlaceCellhelper.centerMassPks1D(states_fixed.Y,param.ampThd);
            pkCenterMass(:,round(i/param.step)) = pkCM;
        elseif DIM == 2
            [pkCM, aveMass] = PlaceCellhelper.centerMassPks(states_fixed.Y,param, param.ampThd); 
            pkCenterMass(:,:,round(i/param.step)) = pkCM;
        else
            disp('Dimension has to be either 1 or 2!')
        end
        pkMas(:,round(i/param.step)) = aveMass;
    end
end
% put all the required output into a structure
output.Yt = Yt;
% output.aveMass = pkCM;
output.pkMas = pkMas;
output.pkAmp = pkAmp;
output.pks = pks;
output.pkCenterMass = pkCenterMass;

end