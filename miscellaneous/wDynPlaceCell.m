% This program explore the dynamics of W when only one place cell

dt = 0.02;
sig = 0.05;
w = rand(param.Ng,1)*0.02;

tot = 1e4;
sep = 10;
w_all = nan(param.Ng, round(tot/sep));
x_ave = mean(gdInput,2);
C = gdInput*gdInput'/1024;
alp = 85;
y = 0;
for i = 1:tot
    y = max(w'*x_ave - alp*y,0);
    w = w + dt*(max(C*w-alp*y*x_ave,0) - w) + sqrt(dt)*sig*randn(param.Ng,1);
    
    % store w
    if mod(i,sep) ==0
        w_all(:,round(i/sep)) = w;
    end
end