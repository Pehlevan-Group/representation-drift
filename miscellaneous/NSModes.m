function dydt = NSModes(t,y,param,X)
% ODEs for solving the NSM problem
dydt =  -y + max((X'*X - param.alpha*ones(param.T,param.T)- param.Q)*y,0)/param.lambda;
end