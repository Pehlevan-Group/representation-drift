function [output, param] = Tmaze_EI_update(X,Xsel,total_iter, param, record_flag)
% runing the population codes, centroid position and noisy weight update and return 
% the learned representations

    ystart = zeros(param.NE,param.BatchSize);  % inital of the output
    zstart = zeros(param.NI,param.BatchSize);  % inital of the inhibitory neurons
    time_points = round(total_iter/param.record_step);
    Yt = nan(param.NE,size(Xsel,2),time_points);
    Zt = nan(param.NI,size(Xsel,2),time_points);
    Num = size(X,2);  % total number of sampling
    
    miniBatch = 10;  % used only for neural dynamics
    numBatch = ceil(size(Xsel,2)/miniBatch);  % total of minibatch

    for i = 1:total_iter
        positions = X(:,randperm(Num,param.BatchSize));
        states= PlaceCellhelper.nsmDynBatchExciInhi(positions,ystart,zstart, param);
        y = states.Y;
        z = states.Z;
        
        % update weight matrix
        param.W = (1-param.learnRate)*param.W + param.learnRate*y*positions'/param.BatchSize + ...
            sqrt(param.learnRate)*param.noise.*randn(param.NE,param.Nx);
        
        param.Wie =max((1-param.learnRate)*param.Wie + param.learnRate*z*y'/param.BatchSize + ...
            sqrt(param.learnRate)*param.noise.*randn(param.NI,param.NE),0);
        
        param.Wei = max((1-param.learnRate)*param.Wei + param.learnRate*y*z'/param.BatchSize + ...
            sqrt(param.learnRate)*param.noise.*randn(param.NE,param.NI),0);
        
        % notice the scaling factor for the recurrent matrix M
        param.M = max((1-param.learnRate)*param.M + param.learnRate*z*z'/param.BatchSize + ...
            sqrt(param.learnRate)*param.noise.*randn(param.NI,param.NI),0);
        
        param.b = (1-param.learnRate)*param.b + param.learnRate*sqrt(param.alpha)*mean(y,2);
        
        if record_flag && mod(i, param.record_step) == 0

            for j = 1:numBatch
                y0 = zeros(param.NE,miniBatch);
                z0 = zeros(param.NI,miniBatch);
                inx = (j-1)*miniBatch+1:min(j*miniBatch,Num);
                states_fixed = PlaceCellhelper.nsmDynBatchExciInhi(Xsel(:,inx),y0,z0, param);
                Yt(:,inx,round(i/param.record_step)) = states_fixed.Y;
                Zt(:,inx,round(i/param.record_step)) = states_fixed.Z;
            end
        end
    end
    output.Yt = Yt;
    output.Zt = Zt;
end

