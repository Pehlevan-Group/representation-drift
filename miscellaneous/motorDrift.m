% this progrma reproduce the single neuron tuning curve drift observed in
% motor cortex in the paper Rokni et al., 2007
% The aim is to see how the drifting dynamics change over time and
% comparing with our Hebbian/anti-Hebbian networks


close all
clear

%% setup and parameters

N = 1e3;   % number of motor neurons
numInput = 1e3;   % number of different angles
totIter = 5e4;    % total iterations
tauLearn = 50;
tauForget = 1500;
sep = 2*pi/numInput;
thetas = sep:sep:2*pi;
X = [cos(thetas);sin(thetas)];

type = 'wildtype';  % simulation type, perturbed or not

W = 0.1*randn(N,2);  % initialization of forward matrix
noiseStd = 0.01;   % noise standard deviation

rndAng = 2*pi*(1/N:1/N:1);
Z = 2/N*[cos(rndAng);sin(rndAng)];  % output matrix to generate force

% initial force
% force = Z*fireRate;   % the force generated

% coding matrix R
if strcmp(type, 'wildtype') 
    R = eye(2);
elseif strcmp(type, 'perturb')
    perturAng = 1/6*pi;  % perturbation angle
    R = [cos(perturAng),-sin(perturAng);sin(perturAng),cos(perturAng)];
end


%% learning and noise
step =  10;   % used to store the peak position
time_points = floor(totIter/step);
pks = nan(N,time_points);    % store all the peak positions
pkAmp = nan(N,time_points);  % store the peak amplitude

storeStep = 5e3;  % store intermediate matrix
Xsel = X(:,10:10:end);  % only store part of the data
numStore = floor(totIter/storeStep);
allW = cell(numStore,1);
wsample = nan(totIter,2);  % store all the weight of example neuron
rateStore = nan(N,numInput/10,numStore);
for i = 1:totIter
    inx = randperm(numInput,1);
    x = X(:,inx); % online learning
    W = W - W/tauForget - N/tauLearn*(Z'*R'*(R*Z*W - eye(2))*x*x') + noiseStd*randn(N,2);
    wsample(i,:) = W(1,:);
    if mod(i, storeStep) == 0
        allW{round(i/storeStep)} = W;
        rateStore(:,:,round(i/storeStep)) = W*Xsel;
    end
    
    % store the peak position (prefered direction)
    if mod(i, step) == 0
        Y = W*X;
        [~,pkInx] = sort(Y, 2,'descend');
        temp = Y((pkInx(:,1)-1)*N + (1:N)');
        pkAmp(:,round(i/step)) = temp;
        pks(:,round(i/step)) = pkInx(:,1);
    end
end


% 
figure
imagesc(rateStore(:,:,1))

figure
imagesc(rateStore(:,:,2))

selNeuron = round(N/2);
figure
hold on
for i = 1:size(rateStore,3)
    y = squeeze(rateStore(selNeuron,:,i));
    plot(y)
end
hold off
Y1 = squeeze(rateStore(:,:,1));
Y2 = squeeze(rateStore(:,:,end));



% Compare the representational similarity
Y1 = rateStore(:,:,1);
Y2 = rateStore(:,:,10);

figure
imagesc(Y1'*Y1)

figure
imagesc(Y2'*Y2)

figure
imagesc(Y2)

% weight matrix over time
W1 = allW{10};
W2 = allW{10};
figure
imagesc(W1'*W1)

figure
imagesc(W2'*W2)

% multi dimensional scaling
% first, distance matrix
D1 = squareform(pdist(Y1','cosine'));
Sc1 = mdscale(D1,3);

figure
plot(Sc1(:,1),Sc1(:,2),'.')

D2= squareform(pdist(Y2','cosine'));
Sc2 = mdscale(D2,3);

figure
plot(Sc1(:,1),Sc2(:,1),'.')

theta1 = atan2(Sc1(:,2),Sc1(:,1));
theta2 = atan2(Sc2(:,2),Sc2(:,1));
figure
plot(theta1,theta2,'o')

%% Estimate the diffusion constant

newPeaks = pks(:,1001:end)/size(X,2)*2*pi;   % 11/21/2020
% newPeaks = pks/size(X,2)*2*pi;   % 11/21/2020

figure
plot(newPeaks([300,700],:)')

msds = nan(floor(time_points/2),N);
for i = 1:floor(time_points/2)
    diffLag = min(abs(newPeaks(:,i+1:end,:) - newPeaks(:,1:end-i,:)),...
        2*pi - abs(newPeaks(:,i+1:end,:) - newPeaks(:,1:end-i,:)) );
    msds(i,:) = nanmean(diffLag.*diffLag,2);
end

Ds = PlaceCellhelper.fitLinearDiffusion(msds,step,'linear');

figure
plot(msds(:,1:5))

% histogram of Ds
figure
histogram(Ds)


%% Compare with offline theory
C = X*X'/N;
Ws = (Z'*Z + eye(N))\(Z'*R'*C);
figure
imagesc(Ws)

figure
plot(W1(:,1),W2(:,1),'.')

figure
plot(W2)