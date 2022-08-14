% this program test the response of trained network on drifting grating
% input

close all
clear

%% setting for the graphics

% plot setting
% defaultGraphicsSetttings
blues = brewermap(13,'Blues');
reds = brewermap(13,'Reds');
greens = brewermap(13,'Greens');
greys = brewermap(11,'Greys');

rb = brewermap(11,'RdBu');
set1 = brewermap(9,'Set1');

spectralMap = flip(brewermap(256,'Spectral'),1);


saveFolder = ['.',filesep,'figures'];

%% noisy non-negative similarity matching

% load the network configuration of the trained network
trainedData = ['.',filesep,'data',filesep,'natMovie_ns1e-05_l1_1e-05_l2_0.0001.mat'];
load(trainedData);

% load the data
load(['.',filesep,'data/gratings.mat']);          % generated gratings
newData = img;
% load('./data/naturalMovieLong.mat');

X = reshape(newData,size(newData,1)*size(newData,2),size(newData,3));
SMX = X'*X;

figure
imagesc(SMX)

%% Check the responses to gratings

Y0 = zeros(dimOut,size(X,2));
Ys = zeros(dimOut,size(X,2));

scaleFactor = 100;   % this is required to observe non-zero response
newX = X*scaleFactor;
% to ensure accuracy,each time run one input
for i = 1:size(X,2)
    [states_fixed_nn, params] = MantHelper.nsmDynBatch(newX(:,i),Y0(:,i),params);
    Ys(:,i) = states_fixed_nn.Y;
end


figure
imagesc(Ys,[0,2.5])
colorbar
xlabel('Frame','FontSize',24)
ylabel('Neuron','FontSize',24)
set(gca,'FontSize',20)

%Figure
figure
plot(Ys(104,:))
xlabel('Frame','FontSize',24)
ylabel('Response','FontSize',24)
set(gca,'FontSize',20,'LineWidth', 1.5)