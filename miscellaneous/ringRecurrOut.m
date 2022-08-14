function Wout = ringRecurrOut(Y,Zhat,kappa)
% this function return the optimal matrix using gradient descent methods
[ny, t] = size(Y); %dimension and number of samples
maxIter = 1e5;
err = inf;
errTol = 1e-5;
count = 1;

% initialize the output matrix by linear part
% covY =Y*Y'/t;
% covYZ = Y*Zhat'/t;
% Wout = pinv(covY + params.kappa*eye(ny))*covYZ;
Wout = randn(ny,size(Zhat,1));

while err > errTol && count <= maxIter
    dW = - 0.05*((Y*(exp(Wout'*Y))' - Y*Zhat')/t + kappa*Wout);
    err = norm(dW)/norm(Wout);
    Wout = Wout + dW;
    count = count + 1;
end
end