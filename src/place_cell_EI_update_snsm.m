function [output, param] = place_cell_EI_update_snsm(X,total_iter, param)
% runing the population codes, centroid position and noisy weight update and return 
% the learned representations
% 
    ystart = zeros(param.Np,param.BatchSize);  % inital of the output
    zstart = zeros(param.Nin,param.BatchSize);  % inital of the inhibitory neurons
    time_points = round(total_iter/param.step);
    Yt = nan(param.Np,param.ps,time_points);
    Zt = nan(param.Nin,param.ps,time_points);

    for i = 1:total_iter
        positions = X(:,randperm(param.ps,param.BatchSize));
        states= PlaceCellhelper.nsmDynBatchExciInhi(positions,ystart,zstart, param);
        y = states.Y;
        z = states.Z;
        
        % update weight matrix
        param.W = (1-param.learnRate)*param.W + param.learnRate*y*positions'/param.BatchSize + ...
            sqrt(param.learnRate)*param.noiseW.*randn(param.Np,param.Ng);
        
        param.Wie =max((1-param.learnRate)*param.Wie + param.learnRate*z*y'/param.BatchSize + ...
            sqrt(param.learnRate)*param.noiseWie.*randn(param.Nin,param.Np),0);
        
        param.Wei = max((1-param.learnRate)*param.Wei + param.learnRate*y*z'/param.BatchSize + ...
            sqrt(param.learnRate)*param.noiseWei.*randn(param.Np,param.Nin),0);
        
        % notice the scaling factor for the recurrent matrix M
        param.M = max((1-param.learnRate)*param.M + param.learnRate*z*z'/param.BatchSize + ...
            sqrt(param.learnRate)*param.noiseM.*randn(param.Nin,param.Nin),0);
        
        param.b = (1-param.learnRate)*param.b + param.learnRate*sqrt(param.alpha)*mean(y,2);
        
        if mod(i, param.step) == 0
            for j = 1:size(X,2)
                y0 = zeros(param.Np,1);
                z0 = zeros(param.Nin,1);
                states_fixed = PlaceCellhelper.nsmDynBatchExciInhi(X(:,j),y0,z0, param);
                Yt(:,j,round(i/param.step)) = states_fixed.Y;
                Zt(:,j,round(i/param.step)) = states_fixed.Z;
            end
        end
    end
    output.Yt = Yt;
    output.Zt = Zt;

end