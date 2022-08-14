function output = generate_place_fields(dim, width, num, sample)
% generate 1D or 2D place fields
% for 1D place fields, we use truncated cosine curve
% for 2D place field, we use 2D gaussian bump

assert(dim == 1 || dim ==2, 'Only support 1D or 2D place field!')
switch dim
    case 1
        % truncated cosine curve, no periodic condition
        output = zeros(num, sample);  % each row is a place field
        thetas = 2*pi/sample*(1:sample) - pi;
        centroids = (1:num)*2*pi/num - pi;
        scaling = pi/2/width;
        for i = 1:num
            acti_index = find(abs(thetas - centroids(i))<=width);
            output(i,acti_index) = max(cos(scaling*(thetas(acti_index) - centroids(i))),0);
        end

    case 2
        % assume square-shaped environment
        output = zeros(num, sample*sample);
        centroids  = lhsdesign(num,2)*2*pi-pi; % x-y coordinates
        thetas = 2*pi/sample*(1:sample) - pi;
        scaling = pi/2/width;
        for i = 1:num
            x_acti = zeros(1,sample);
            y_acti = zeros(1,sample);
            acti_index_x = abs(thetas - centroids(i,1))<=width;
            acti_index_y = abs(thetas - centroids(i,2))<=width;
            x_acti(acti_index_x) = max(cos(scaling*(thetas(acti_index_x) - centroids(i,1))),0);
            y_acti(acti_index_y) = max(cos(scaling*(thetas(acti_index_y) - centroids(i,2))),0);
            temp = x_acti'*y_acti;
            output(i,:) = temp(:);
        end
      
end