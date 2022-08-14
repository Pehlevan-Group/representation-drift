function y = nonNegativeProjection(X,W,M)
% this function return the stedy state value of ys using projection
% gradient
err = 1e10;
err_tol = 1e-3;
max_iter = 1e3;
count = 0;
yold = zeros(size(M,1),size(X,2));
while err > err_tol && count < max_iter
    y = max(pinv(M)*W*X,0);
    err = norm(y-yold,'fro');
    yold = y;
    count = count + 1;
end
end