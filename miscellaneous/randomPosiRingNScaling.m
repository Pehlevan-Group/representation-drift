N = [3,5,10,20,50,100];
repeat = 100;

aveDist = nan(length(N),repeat);
for i = 1:length(N)
    rps = rand(N(i),repeat);
    [ps,inx] = sort(rps,1);
    temp = [diff(ps,1,1);ps(1,:) - ps(end,:) + 1];
    aveDist(i,:) = std(temp(:)*2*pi,0,1);
end

figure
plot(N,mean(aveDist,2))
set(gca,'YScale','log','XScale','log')