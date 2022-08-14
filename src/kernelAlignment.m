function align = kernelAlignment(K1,K2)
% K1 , K2 are N by T response matrices
% return the kernel alignment
% align = trace((K1'*K1)*(K2'*K2))/sqrt(trace((K1'*K1)*(K1'*K1))*trace((K2'*K2)*(K2'*K2)));
align = norm(K1*K2','fro')^2/norm(K1'*K1,'fro')/norm(K2'*K2,'fro');
end