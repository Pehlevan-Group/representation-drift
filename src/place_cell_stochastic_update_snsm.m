function [output, param] = place_cell_stochastic_update_snsm(X,total_iter,...
    param, slow_flag, storage_flag, input_flag)
% runing the population codes, centroid position and noisy weight update and return 
% the learned representations
% 
if exist('slow_flag','var')
    Flag = slow_flag;
else
    Flag = false;
end

if exist('storage_flag','var')
    Flag_store = storage_flag;
else
    Flag_store = false;
end

if exist('input_flag','var')
    Flag_input = input_flag;
    % generate background noise
    gridFields_bg= background_grid_1d(param);
    num_bg = size(gridFields_bg,1);  
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
    if ~Flag
        if Flag_input
            states = PlaceCellhelper.nsmDynBatch(positions,ystart, param);
            y = states.Y;
        else
            % use background noise
            positions =  gridFields_bg(randperm(num_bg,1),:);
            states = PlaceCellhelper.nsmDynBatch(positions',ystart, param);
            y = states.Y;
%             y = zeros(param.Np,param.BatchSize);
        end
        % noisy update weight matrix
        param.W = (1-param.learnRate)*param.W + param.learnRate*y*positions'/param.BatchSize + ...
            sqrt(param.learnRate)*param.noiseW.*randn(param.Np,param.Ng); 
        param.M = max(0,(1-param.learnRate)*param.M + param.learnRate*y*y'/param.BatchSize + ...
            sqrt(param.learnRate)*param.noiseM.*randn(param.Np,param.Np));
        param.b = (1-param.learnRate)*param.b + param.learnRate*sqrt(param.alpha)*mean(y,2);
        
    elseif Flag
        if Flag_input
            states = PlaceCellhelper.nsmDynBatchMultiScale(positions,ystart, param);
            y = states.Y;
        else
            % use background noise
            positions =  gridFields_bg(randperm(num_bg,1),:)';
            states = PlaceCellhelper.nsmDynBatch(positions,ystart, param);
            y = states.Y;
%             y = zeros(param.Np,param.BatchSize);
        end
        % update the slow synapses
        param.Wslow = (1-param.learnSlow)*param.Wslow + Flag_input*param.learnSlow*y*positions'/param.BatchSize + ...
        sqrt(param.learnSlow)*param.noiseW.*randn(param.Np,param.Ng); 
        param.Mslow = max(0,(1- param.learnSlow)*param.Mslow + Flag_input*param.learnSlow*y*y'/param.BatchSize + ...
        sqrt(param.learnSlow)*param.noiseM.*randn(param.Np,param.Np));
        
        param.W = (1-param.learnRate)*param.W + Flag_input*param.learnRate*y*positions'/param.BatchSize + ...
            sqrt(param.learnRate)*param.noiseW.*randn(param.Np,param.Ng); 
        param.M = max(0,(1-param.learnRate)*param.M + Flag_input*param.learnRate*y*y'/param.BatchSize + ...
        sqrt(param.learnRate)*param.noiseM.*randn(param.Np,param.Np));
        param.b = (1-param.learnRate)*param.b + param.learnRate*sqrt(param.alpha)*mean(y,2);
    end
    
    
    % store and check representations
    if mod(i, param.step) == 0 && Flag_store
        Y0 = zeros(param.Np,size(X,2));
        if Flag
            states_fixed = PlaceCellhelper.nsmDynBatchMultiScale(X,Y0, param);
        else
%             states_fixed = PlaceCellhelper.nsmDynBatch(X,Y0, param);
            Ys = zeros(param.Np,size(X,2));
            for idx = 1:size(X,2) 
                states= PlaceCellhelper.nsmDynBatch(X(:,idx),Y0(:,idx), param);
                Ys(:,idx) = states.Y;
            end
            states_fixed.Y = Ys; % this is just for notational consistence
        end
        
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