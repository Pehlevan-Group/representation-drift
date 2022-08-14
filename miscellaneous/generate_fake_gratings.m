x = 0:0.01:50; 
t = 30;             % number of timepoints to generate for each ori, default 50
sf = 30;             % spatial frequency (arbitrary units)
fr = 30;            % framerate, we can also down scale this 
h = length(x);            % image height (arbitrary)
num_ori = 12;

% we can make the image smaller if we want, default is 500 x 500
% img = zeros(50, 50, t*num_ori);           % height x width x time
width = 54; height = 96;           % the same size as in the natural movie
img = zeros(width, height, t*num_ori);           % height x width x time

for ori = 1:12
    fprintf('generate ori %d of %d\n', ori, num_ori);
    for ii = 1:t
        y = sin(sf*x);             
        y_full = repmat(y, h, 1);
        y_full = imrotate(y_full, (ori-1)*(360/num_ori));           % each orientation is 30 deg apart
        img(:, :, (ori-1)*t+ii) = y_full(3001:3000+width, 3001:3000+height);   
        x = x + 0.15;           
    end
end
%%

%playback
figure; colormap gray;
for ii = 1:size(img, 3)
    imagesc(img(:, :, ii));
    pause(1/fr);                    
end


%% normalize the responses
Z = reshape(img, width*height,t*num_ori);
X = Z'*Z;    %input similarity matrix

figure
imagesc(X)
colorbar

max(X(:))
img = img/sqrt(max(X(:)));

figure
imagesc(reshape(img,width*height,t*num_ori));
colorbar

Z2 = reshape(img,width*height,t*num_ori);
figure
imagesc(Z2'*Z2)
colorbar