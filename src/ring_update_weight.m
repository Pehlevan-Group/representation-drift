function [Yt, params] = ring_update_weight(X,total_iter,params,input_flag)
% whether or not with input
if exist('input_flag','var')
    Flag = input_flag;
else
    Flag = true;
end

% update the synaptic weights and the stored population vectors if required
time_points = round(total_iter/params.record_step);
Xsel = X(:,1:4:end);
Yt = nan(params.dim_out,size(Xsel,2),time_points);
num_samp = size(X,2);

    for i = 1:total_iter
        if Flag
            y0 = 0.1*rand(params.dim_out,params.batch_size);
            inx = randperm(num_samp,params.batch_size);
            x = X(:,inx);  % randomly select one input
    %         y = MantHelper.quadprogamYfixed(x,params);
            states= MantHelper.nsmDynBatch(x,y0, params);
            y = states.Y;
        else
            x = zeros(params.dim_in,1);
            y = zeros(params.dim_out,1);
        end
        % update weight matrix
        params.W = (1-params.learnRate)*params.W + params.learnRate*y*x'/params.batch_size...
            +sqrt(params.learnRate)*params.noiseW.*randn(params.dim_out,params.dim_in);        
        params.M = max((1-params.learnRate)*params.M + params.learnRate*y*y'/params.batch_size + ...
            sqrt(params.learnRate)*params.noiseM.*randn(params.dim_out,params.dim_out),0);
        params.b = (1-params.learnRate)*params.b + params.learnRate*sqrt(params.alpha)*mean(y,2);
        
        % store every param.step steps
        if mod(i, params.record_step) == 0
            y0 = zeros(params.dim_out,size(Xsel,2));
            states_fixed = MantHelper.nsmDynBatch(Xsel,y0, params);
            Yt(:,:,round(i/params.record_step)) = states_fixed.Y;
        end
        
    end
end