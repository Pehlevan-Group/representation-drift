function [output, param] = place_cell_stochastic_update_forget(X,total_iter,...
    param, storage_flag, input_flag)
% runing the population codes, centroid position and noisy weight update and return 
% the learned representations

if exist('storage_flag','var')
    Flag_store = storage_flag;
else
    Flag_store = false;
end

if exist('input_flag','var')
    Flag_input = input_flag; 
else
    Flag_input = true;
end


ystart = zeros(param.Np,param.BatchSize);  % inital of the output
time_points = round(total_iter/param.step); 
Yt = nan(param.Np,param.ps,time_points);
pkMas = nan(param.Np,time_points);  
pkAmp = nan(param.Np,time_points); 
pks = nan(param.Np,time_points);
pkCenterMass = nan(param.Np,time_points);

% generate position code by the grid cells    
for i = 1:total_iter
    positions = X(:,randperm(param.ps,param.BatchSize));
   
    if Flag_input
        states = PlaceCellhelper.nsmDynBatch(positions,ystart, param);
        y = states.Y;
    else
         y = zeros(param.Np,param.BatchSize);
    end

    param.W = (1-param.forgetRate)*param.W + sqrt(param.forgetRate)*param.noiseW.*randn(param.Np,param.Ng) + ...
        Flag_input*(param.learnRate*(y*positions'/param.BatchSize - param.W) + ...
        sqrt(param.learnRate)*param.noiseW.*randn(param.Np,param.Ng)); 
    param.M = max(0,(1-param.forgetRate)*param.M  + sqrt(param.forgetRate)*param.noiseM.*randn(param.Np,param.Np) + ...
        Flag_input*(param.learnRate*(-param.M + y*y'/param.BatchSize) + ...
        sqrt(param.learnRate)*param.noiseM.*randn(param.Np,param.Np)));
    param.b = (1-param.forgetRate)*param.b + Flag_input*param.learnRate*(sqrt(param.alpha)*mean(y,2) - param.b);
%     param.b = param.b + Flag_input*param.learnRate*(sqrt(param.alpha)*mean(y,2) - param.b);

    % store and check representations
    if mod(i, param.step) == 0 && Flag_store
        Y0 = zeros(param.Np,size(X,2));
        states_fixed = PlaceCellhelper.nsmDynBatch(X,Y0, param);
        
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
output.pkMas = pkMas;
output.pkAmp = pkAmp;
output.pks = pks;
output.pkCenterMass = pkCenterMass;

end