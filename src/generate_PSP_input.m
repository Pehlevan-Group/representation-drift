function [X, Xtest, V] = generate_PSP_input(input_cov_eigens,input_dim,output_dim, num_sample)

% generate input data, first covariance matrix
V = orth(randn(input_dim,input_dim));
U = orth(randn(output_dim,output_dim));       % arbitrary orthogonal matrix to determine the projection
% C = V*diag(input_cov_eigens)*V'/sqrt(input_dim);    % with normalized length
C = V*diag(input_cov_eigens)*V';    % with normalized length
Vnorm = norm(V(:,1:output_dim));

% generate multivariate Gaussian distribution with specified correlation
% matrix
X = mvnrnd(zeros(1,input_dim),C,num_sample)';
Cx = X*X'/num_sample;      % input sample covariance matrix

% generate some test stimuli that are not used during learning
num_test = 1e2;
Xtest = mvnrnd(zeros(1,input_dim),C,num_test)';

end