%%Use ONLINE PCA with Hebbian Anti-Hebbian Learning rule (Pehlevan et. al,2015 Neural Computation)
clear all
dFoler = 'C:\Users\shq382\Documents\data'; 
fName = 'SortedDataSample_Ruster_560_1800000.bin';
fileID = fopen(fullfile(dFoler,fName));
raw = fread(fileID);

% first offline PCA
% [coeff,score,latent] = pca(raw);

% using online fast similarity matching
d=560;  % input dimensionality
k=20;   % output dimensionality
n=1.8e6; % number of total data point
raw = reshape(raw,[d,n]);

% fast SM
% Uhat0 = bsxfun(@times,raw( :,1:k), 1./sqrt(sum(raw(:,1:k).^2,1)))';
% scal = 100;
% Minv0 = eye(k) * scal;
% Uhat0 = Uhat0 / scal;
fsm = FSM(k, d, [], [], [], []);
psp_comp = nan(k,n);
for i = 1:n
    psp_comp(:,i)= fsm.fit_next(raw(:,i)');
end


% visualize the lower dimensional manifold
% plot the first 3d
ns = 18000;
tt = psp_comp(:,1:ns);
traw = raw(:,1:ns);  % raw sample
figure
plot((1:ns)',traw(2,:)')


figure
plot3(tt(1,:)',tt(2,:)',tt(3,:)')

method = 'spiked_covariance_normalized';
options_generator=struct;
options_generator.rho=0.01;
options_generator.lambda_q=1;
[x,eig_vect,eig_val] = low_rank_rnd_vector(d,k,n,method,options_generator);
[x,eig_vect,eig_val] = standardize_data(x,eig_vect,eig_val);
disp('generated')
%%
errors=zeros(n,1)*NaN;
Uhat0 = bsxfun(@times,x( :,1:k), 1./sqrt(sum(x(:,1:k).^2,1)))';
scal = 100;
Minv0 = eye(k) * scal;
Uhat0 = Uhat0 / scal;
fsm = FSM(k, d, [], Minv0, Uhat0, []);
psp_comp = nan(k,n);
for i = 1:n
    if mod(i,50) == 1
        errors(i,:) = compute_projection_error(eig_vect, fsm.get_components([]));
        disp(i)
    end
    psp_comp(:,i)= fsm.fit_next(x(:,i)');
end
loglog(1:n,errors,'.')