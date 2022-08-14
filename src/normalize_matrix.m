function A_normalized = normalize_matrix(A, a_norm)
% This function normlizes the rows of A
% A: the matrix to be normalized
% a_norm: norm

% Set defaut value for 'aNorm'
if ~exist('a_norm', 'var')
    a_norm = 1;
end

% upper_bound = 1

% A small value to avoid zero division
epsilon = 1e-15;

% Normalize A

% Normalize each row to l2-norm
A_normalized = diag( a_norm ./ ( sqrt(sum(A.*A,2)) + epsilon ))*A;


end