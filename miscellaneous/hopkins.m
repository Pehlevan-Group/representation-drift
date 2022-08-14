% Created: 12/30/2012
% Created by: G. Matthew Fricke
% Modified: 12/30/2012
% Modified by: G. Matthew Fricke
% Version: 1.0
%
% Copyright: Freely Distributable
%
% Description. Characterizes how clustered a group of input points are by
% comparing the points to a group of uniformly distribted points.
% The Hopkins Statistic is the ratio of the sums of the nearest neighbor
% distances for the input data and uniformly distributed points.
% In this varient (a version of the Normalized Hopkins Statistic) the
% closer the statistic is to one half the less clustered the input points,
% the closer to one the more clustered the points are.
%
% This code follows the Hopkins Statistic variant described in Zhang J., 
% Leiderman K., Pfeiffer J.R., Wilson B.S., Oliver J.M., Steinberg S.L., 
% "Characterizing the topography of membrane receptors and signaling 
% molecules from spatial patterns obtained using nanometer-scale
% electron-dense probes and electron microscopy." Micron. 2006;37(1):14-34.
% Epub 2005 Jul 11. in turn based on the statistic descibed in Hopkins, B. 
% (1954) A new method for determining the type of distribution of plant 
% individuals. Ann Bot.(London) 18:213-227.
%
% Note: This code is parallelized. The outermost for loop is a parloop
% since the different replicantions have no interdependency.
%
% Input Parameters:
% d is a handle to the distance function to use
% S is the set of input points represented as x, y coordinates columns
% m is the sample size to use each iteration (repetitions), i.e the number
% of points for which the nearest neighbor distance in S is found.
% Assumption m << |S|
% r is the number of replications to perform
%
% Output:
% a histogram showing the binned Hopkins Statisitic over all the replications. 
% a vector containing the Hopkins Statistic for each of the replications
% the further the value of the Hopkins Statistic is away from one
% half the more the input distribution is skewed from uniform. Values
% greater than one half are increasingly clustered. The range of possible
% values is 0...1.
%
% Example: hopkins(@(p1, p2) sqrt(sum((p1-p2).^2)), randi(100, 200, 2), 10, 1000)
% Where randi produces some x, y coordinate columns and @(p1, p2)... is the euclidean distance

function [mean_reps,std_reps] = hopkins( d, S, m, r )
 
% Check arguments
[rows, cols] = size(S);
 
if rows < 1
    error('The input should include at least one point');
end
 
if cols ~= 2
    error('The input points should be in form of an n x 2 matrix. The column is a list of x coordinates and the second is a list of y coordinates. Called with input matrix of size %d x %d.', rows, cols);
end
 
if m < 1
    error('The sample size for the Hopkins Statistic must be at least one. Called with %d.', m);
end
 
if r < 1
    error('The number of replicants to run must be at least one. Called with %d.', r);
end
 
tic % Time the function
 
% Create a pool of parallel computation workers 
% if matlabpool('size') == 0 % checking to see if my pool is already open
%     matlabpool open 
% end
 
% Find the limits of S, i.e. the maximum and minumum x and y coordinates
max_x = max(S(:,1));
max_y = max(S(:,2));
min_x = min(S(:,1));
min_y = min(S(:,2));
 
len = length(S);
 
replications = zeros(1,r);
parfor reps = 1:r
    distances = zeros(1,len); % preallocate space
        
    % Generate a uniformly distributed set of m points within the limits of S
    R_x = min_x + (max_x-min_x)*rand(m,1);
    R_y = min_y + (max_y-min_y)*rand(m,1);
    R = [R_x, R_y];
    
    plot (R, 'x');
    
    
    % For m randomly chosen points t_i in S
    % Find the nearest neighbor to t_i in {S-t_i} and calculate the distance x_i.
    % Let W equal the sum the distances x_i
    n_points = length(S);
    indices = randi(n_points, m, 1);
    W = 0;
    for i = 1:m
        t = S(indices(i),:);
        for j = 1:len
            distances(j) = d(t,S(j,:))^2;
        end
        
        distances(indices(i)) = []; % Remove the distance from t to t
        nn_dist = min(distances); % Nearest neighbor distance to t
        W = W + nn_dist; % Keep track of the sum of nearest neighbor distances
    end
    
    % For m randomly chosen points r_i in R
    % Find the nearest neighbor to r_i in {S-r_i} and calculate the distance y_i.
    % Let U equal the sum the distances y_i
    U = 0;
    for i = 1:m
        u = R(i,:);
        for j = 1:len
            distances(j) = d(u,S(j,:))^2;
        end
        
        nn_dist = min(distances); % Nearest neighbor distance to r
        U = U + nn_dist; % Keep track of the sum of nearest neighbor distances
    end
    
    % Calculate the normalized Hopkins statistic H
%     H = U/(U+W);
    H = W/U;
    
    % Add the Hopkins statistic for this experiment to an array
    replications(reps) = H;
end
 
% Display a histogram of the results for the r replications.
% histogram(replications, (0.0:0.01:0.99)+0.005);
histogram(replications);

 
mean_reps = mean(replications);
std_reps = std(replications);

toc % End timer
 
end