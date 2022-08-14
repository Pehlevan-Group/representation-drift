function [output, param] = ring_update_multi_timescale(X,total_iter,param, input_flag)
% runing the population codes, centroid position and noisy weight update and return 
% the learned representations
% 
if exist('input_flag','var')
    Flag = input_flag;
else
    Flag = true;
end

ystart = zeros(param.Np,param.batch_size);  % inital of the output
time_points = round(total_iter/param.step);
Yt = nan(param.Np,param.ps,time_points);
pkMas = nan(param.Np,time_points);  
pkAmp = nan(param.Np,time_points); 
pks = nan(param.Np,time_points);
pkCenterMass = nan(param.Np,time_points);

% generate position code by the grid cells  
for i = 1:total_iter
    if Flag
        positions = X(:,randperm(param.ps,param.batch_size));
        states = PlaceCellhelper.nsmDynBatchMultiScale(positions,ystart, param);
        y = states.Y;
        % noisy update weight matrix
        param.W = (1-param.learnRate)*param.W + param.learnRate*y*positions'/param.batch_size + ...
            sqrt(param.learnRate)*param.noiseW.*randn(param.Np,param.dim_in);
        param.Wslow = (1-param.learnSlow)*param.Wslow + param.learnSlow*y*positions'/param.batch_size + ...
            sqrt(param.learnSlow)*param.noiseW.*randn(param.Np,param.dim_in);
        param.M = max(0,(1-param.learnRate)*param.M + param.learnRate*y*y'/param.batch_size + ...
            sqrt(param.learnRate)*param.noiseM.*randn(param.Np,param.Np));
        param.Mslow = max(0,(1-param.learnSlow)*param.Mslow + param.learnSlow*y*y'/param.batch_size + ...
            sqrt(param.learnSlow)*param.noiseM.*randn(param.Np,param.Np));
        param.b = (1-param.learnRate)*param.b + param.learnRate*sqrt(param.alpha)*mean(y,2);
    else
        y = zeros(size(ystart));
        % noisy update weight matrix
        
        param.W = (1-param.learnRate)*param.W + sqrt(param.learnRate)*param.noiseW.*randn(param.Np,param.dim_in);
        param.Wslow = (1-param.learnSlow)*param.Wslow + sqrt(param.learnSlow)*param.noiseW.*randn(param.Np,param.dim_in);
        param.M = max(0,(1-param.learnRate)*param.M + sqrt(param.learnRate)*param.noiseM.*randn(param.Np,param.Np));
        param.Mslow = max(0,(1-param.learnSlow)*param.Mslow + sqrt(param.learnSlow)*param.noiseM.*randn(param.Np,param.Np));
        param.b = (1-param.learnRate)*param.b + param.learnRate*sqrt(param.alpha)*mean(y,2);
    end

    

    % store and check representations
    if mod(i, param.step) == 0
        Y0 = zeros(param.Np,size(X,2));
        states_fixed = PlaceCellhelper.nsmDynBatchMultiScale(X,Y0, param);
        flag = sum(states_fixed.Y > param.ampThd,2) > 3;   % only those neurons that have multiple non-zeros considered
        [~,pkInx] = sort(states_fixed.Y,2, 'descend');
        temp = states_fixed.Y((pkInx(:,1)-1)*param.Np + (1:param.Np)');
        pkAmp(flag,round(i/param.step)) = temp(flag);
        pks(flag,round(i/param.step)) = pkInx(flag,1);
        Yt(:,:,round(i/param.step)) = states_fixed.Y;

        % store the center of mass
        [pkCM, aveMass] = PlaceCellhelper.centerMassPks1D(states_fixed.Y,param.ampThd);
        pkCenterMass(:,round(i/param.step)) = pkCM;
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