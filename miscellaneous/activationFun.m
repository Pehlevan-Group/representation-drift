function ys = activationFun(X,W,M,actiType)
    % return the fixed point
    if strcmp(actiType,'linear')
        ys = pinv(M)*W*X;
    elseif strcmp(actiType,'relu')
        ys = nonNegativeProjection(X,W,M);
    end
end