function df = dphi(x)
% the deriviative of phi(x), modulo of 2pi
L = length(x);
df = ones(L,1);
indx1 = x > pi;
indx2 = x > -pi & x <=0;
df(indx1 | indx2) = -1;
end