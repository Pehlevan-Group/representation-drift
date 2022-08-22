function [Yt, param] = place_cell_learn_forget(gdInput,num_session, param)
% update the weight during training (with sensory input) and forgetting
% during absence of sensory input, data used for Fig S7

iter_in_session = round((param.learnTime + param.forgetTime)/param.step);  % number of iteration in one session
iter_in_learn = round(param.learnTime/param.step);
iter_in_forget = round(param.forgetTime/param.step);
tot_iteration = iter_in_session*num_session;

% ystart = zeros(param.Np,param.BatchSize);  % inital of the output
Yt = nan(param.Np,param.ps,tot_iteration);

% pretraining process
[~, param] = place_cell_stochastic_update_forget(gdInput,param.preTrain, param,false, true);  

 for i = 1:num_session
    
  % learning part
  [output, param] = place_cell_stochastic_update_forget(gdInput,param.learnTime, param,true, true);  
  Yt(:,:, (i-1)*(iter_in_session)+1:(i-1)*(iter_in_session)+iter_in_learn) = output.Yt;
  
  % forgetting part
  [output, param] = place_cell_stochastic_update_forget(gdInput,param.forgetTime, param,true, false);  
  Yt(:,:, i*iter_in_session-iter_in_forget+1:i*iter_in_session) = output.Yt;
  
end 
end
