classdef MantHelper < handle
    
    properties
        
    end
    
    methods(Static)
        
        % project gradient descent, as in manifold tiling paper
        function [states, params] = neuralDynOnline(x,y,z,V, params)
            MaxIter = 1e4; % maximum iterations
            ErrTol = 1e-6; % error tolerance
            count = 0;
            err = inf;
            yold = y;
            zold = z;
            
            while count < MaxIter && err > ErrTol
                y = max(y + params.gy*(params.W*x - V'*z - sqrt(params.alpha)*params.b), 0);
                z = max(z + params.gz*(-params.beta*z + V*y),0);
                V = max(V + params.gv*(z*y' - V),0);
                err = norm(y-yold,'fro')/(norm(yold,'fro')+ 1e-10) + norm(z-zold,'fro')/(norm(zold,'fro')+ 1e-10);
                yold = y;
                zold = z;
                count = count + 1;
            end
            states.y = y;
            states.z = z;
            states.V = V;
        end
        
        % offline neural dynamics
        function [states, params] = neuralDynBatch(X,Y,Z,V, params)
            MaxIter = 1e4; % maximum iterations
            ErrTol = 1e-5; % error tolerance
            count = 0;
            err = inf;
            Yold = Y;
            Zold = Z;
            T = size(X,2);  % number of samples
            
            while count < MaxIter && err > ErrTol
                Y = max(Y + params.gy*(params.W*X - V'*Z - sqrt(params.alpha)*params.b), 0);
                Z = max(Z + params.gz*(-params.beta*Z + V*Y),0);
                V = max(V + params.gv*(Z*Y'/T - V),0);
                err = norm(Y-Yold,'fro')/(norm(Yold,'fro')+ 1e-10)/T + norm(Z-Zold,'fro')/(norm(Zold,'fro')+ 1e-10)/T;
                Yold = Y;
                Zold = Z;
                count = count + 1;
            end
            states.Y = Y;
            states.Z = Z;
            states.V = V;
        end
        
        
        % simple non-negative similarity matching neural dynamics
        function states = nsmDynBatch(X,Y, params)
            MaxIter = 1e4; % maximum iterations
            ErrTol = 1e-4; % error tolerance, should be smaller for convergence
            count = 0;
            err = inf;
            cumErr = inf;
%             uyold = zeros(size(Y));
            m = mean(diag(params.M));
            uyold = Y*(params.lbd2 + m);   % revised on 05/11/2021
            Yold = Y;
            T = size(X,2);  % number of samples
            errTrack = rand(5,1); % store the lated 5 error
            cumErrTol = 1e-10;    % 1e-6
            
            % using decaying learning rate
            while count < MaxIter && err > ErrTol && cumErr > cumErrTol
%                 dt = max(params.gy/(1+count/10),1e-2);
                dt = params.gy;
                uy = uyold + dt*(-uyold + params.W*X - sqrt(params.alpha)*params.b - ...
                    (params.M - diag(diag(params.M)))*Yold);
                Y = max((uy - params.lbd1)./(params.lbd2 + diag(params.M)), 0);
                err = norm(Y-Yold,'fro')/(norm(Yold,'fro')+ 1e-12)/dt;
                Yold = Y;
                uyold = uy;
                count = count + 1;
                errTrack = [err;errTrack(1:4)];
                cumErr= abs(sum(diff(errTrack)));
            end
            states.Y = Y;
        end
        
        % estimate the centroid of the ring model
        function centroid = nsmCentroidRing(Y, flags)
            [num,bins] = size(Y);  % number of neurons and bins of 2pi rnage
            half_bins = round(bins/2);
            centroid = nan(num,1);
            thd = 0.1;    % default threshod to remove small bumps
            for i = 1:num
                if Y(i,end) > 0.01 && flags(i)
                    temp = [Y(i,:),Y(i,:)];
                    temp(temp<thd) = 0;   % remove small bump
                    cm = mean(temp(half_bins:3*half_bins).*(half_bins:3*half_bins))/mean(temp(half_bins:3*half_bins));
                    if cm > bins
                        centroid(i) = cm - bins;
                    else
                        centroid(i) = cm;
                    end  
                elseif Y(i,end) <= 0.01 && flags(i)
                    temp = Y(i,:);
                    temp(temp <thd) = 0;   % remove small bump
                    centroid(i) = mean(temp.*(1:bins))/mean(temp); 
                end
            end
        end
        % simple non-negative similarity matching neural dynamics
        function [states, params] = nsmRingModel(X,Y, params)
            MaxIter = 1e4; % maximum iterations
            ErrTol = 1e-3; % error tolerance
            count = 0;
            err = inf;
            cumErr = inf;
            uyold = zeros(size(Y));
            Yold = Y;
            T = size(X,2);  % number of samples
            errTrack = rand(5,1); % store the lated 5 error
            cumErrTol = 1e-6;
            
            % using decaying learning rate
            while count < MaxIter && err > ErrTol && cumErr > cumErrTol
%                 dt = max(params.gy/(1+count/10),1e-2);
                dt = params.gy;
                uy = uyold + dt*(-uyold + params.W*X - sqrt(params.alpha)*params.b - ...
                    (params.M - diag(diag(params.M)))*Yold);
%                 Y = max((uy - params.lbd1)./(params.lbd2 + diag(params.M)), 0);
                Y = max((uy - params.lbd1), 0);
                err = norm(Y-Yold,'fro')/(norm(Yold,'fro')+ 1e-10)/T/dt;
                Yold = Y;
                uyold = uy;
                count = count + 1;
                errTrack = [err;errTrack(1:4)];
                cumErr= abs(sum(diff(errTrack)));
            end
            states.Y = Y;
        end
        
        % with autopase
        % simple non-negative similarity matching neural dynamics
        function [states, params] = nsmDynAutopase(X,Y, params)
            MaxIter = 1e4; % maximum iterations
            ErrTol = 1e-3; % error tolerance
            count = 0;
            err = inf;
            cumErr = inf;
            uyold = zeros(size(Y));
            Yold = Y;
            T = size(X,2);  % number of samples
            errTrack = rand(5,1); % store the lated 5 error
            cumErrTol = 1e-6;
            
            % using decaying learning rate
            while count < MaxIter && err > ErrTol && cumErr > cumErrTol
%                 dt = max(params.gy/(1+count/10),1e-2);
                dt = params.gy;
                uy = uyold + dt*(-uyold + params.W*X - sqrt(params.alpha)*params.b - ...
                    params.M*Yold);
                Y = max(uy - params.lbd1, 0);
                err = norm(Y-Yold,'fro')/(norm(Yold,'fro')+ 1e-10)/T/dt;
                Yold = Y;
                uyold = uy;
                count = count + 1;
                errTrack = [err;errTrack(1:4)];
                cumErr= abs(sum(diff(errTrack)));
            end
            states.Y = Y;
        end

        
        % simple non-negative similarity matching neural dynamics with
        % current noise
        function [states, params] = nsmNoiseDynBatch(X,Y, params)
            MaxIter = 1e4; % maximum iterations
            ErrTol = 1e-3; % error tolerance
            count = 0;
            err = inf;
            cumErr = inf;
            uyold = zeros(size(Y));
            Yold = Y;
            T = size(X,2);  % number of samples
            errTrack = rand(5,1); % store the lated 5 error
            cumErrTol = 1e-6;
            
            % using decaying learning rate
            while count < MaxIter && err > ErrTol && cumErr > cumErrTol
%                 dt = max(params.gy/(1+count/10),1e-2);
                dt = params.gy;
                uy = uyold + dt*(-uyold + params.W*X - sqrt(params.alpha)*params.b - ...
                    (params.M - diag(diag(params.M)))*Yold);
                Y = max((uy - params.lbd1)./(params.lbd2 + diag(params.M)), 0);
                err = norm(Y-Yold,'fro')/(norm(Yold,'fro')+ 1e-10)/T/dt;
                Yold = Y;
                uyold = uy;
                count = count + 1;
                errTrack = [err;errTrack(1:4)];
                cumErr= abs(sum(diff(errTrack)));
            end
            
            % add current noise
            uy = uy + randn(size(uy))*params.noise;
            states.Y = max((uy - params.lbd1)./(params.lbd2 + diag(params.M)), 0);
        end
        
        % estimate the fixed point using quadratic prgraming
        function ys = quadprogamYfixed(x,params)
            dimOut = size(params.M,1);
            Mbar = params.M + eye(dimOut)*params.lbd2;
            Q = -(params.W*x - params.alpha*params.b); % revised on 04/28/2021
            ys = quadprog(Mbar,Q,-eye(dimOut),zeros(dimOut,1));
        end
        
        % vanina quadratic programming for NSM0
        function Ys = quadprogmNSM0(X,Y,alp,lbd)
            Q = Y'*Y;
            D = X'*X;
            [N,T] = size(Y);
            H = max(D - alp*eye(T) - Q)/lbd;
            M = H'*H - 2*H + eye(T);
            Ys = nan(N,T);
            for i = 1:N
                ys = quadprog(M,zeros(T,1),-eye(T),zeros(T,1));
                Ys(i,:) = ys';
            end
        end
        
        % Quadratic programming to solve eq 43 in manifold tiling paper
        function ys = quadprogramNSM(X,Y,params)
            D = X'*X;
            Q = Y'*Y;
            Q_hat = D-params.alpha*ones(params.T) - Q;
            Q_tilde = (Q_hat- params.lambda*eye(params.T));
            H = Q_tilde'*Q_tilde;
            ys = quadprog(H,zeros(params.T,1),-Q_hat,zeros(params.T,1),[],[],zeros(params.T,1));
        end
        
        % Least square programming to solve the NSM problems
        function ys = lsqlinNSM(X,Y,params, y0)
            options = optimoptions('lsqlin','Algorithm','trust-region-reflective','Display','iter');
            D = X'*X;
            Q = Y'*Y;
            Q_hat = D-params.alpha*ones(params.T) - Q;
            C = Q_hat- params.lambda*eye(params.T);
            ys = lsqlin(C,zeros(params.T,1), [],[],[],[], zeros(params.T,1), [],y0, options);
        end
        
        % Nonlinear least square optimization
        function ys = lsqnonlinNSM(X,Y,params, y0)
            D = X'*X;
            Q = Y'*Y;
            Q_hat = D-params.alpha*ones(params.T) - Q;
%             C = Q_hat- params.lambda*eye(params.T);
            myfun = @(y) (max(Q_hat*y - params.lambda)'*max(Q_hat*y - params.lambda));
            options = optimoptions(@lsqnonlin, 'Algorithm','trust-region-reflective','Display','iter');
            ys = lsqnonlin(myfun,y0,zeros(params.T,1),[],options);
        end
        
        % iteration method for solving the NSM problems
        function ys = iterationProjNSM(X,y0,params, step)
%             Y = toeplitz(y0,y0);
            shift = floor(length(y0)/params.N);
            Y = MantHelper.circShiftVec(y0,shift)/sqrt(params.N);
            D = X'*X;
            Q = Y'*Y;
%             Q_hat = D-params.alpha*ones(params.T) - Q;
            
            maxIter = 1e5;  % maximum number of iterations
            errTol = 1e-6;  % tolarance of error
            err = inf;      % intinial error
            count = 1;
            yold = 0.5*cos((1:params.T)'*2*pi/params.T - pi);
            yold(yold<0) = 0;
            while(count < maxIter && err > errTol)
%                 ys = yold + step*1/params.lambda*max(Q_hat*yold,0);
%                 ys = yold + step*(-params.lambda*yold + max(Q_hat*yold,0));
                ys = (1-step*params.lambda)*yold + step*max((D -params.alpha*ones(params.T)-Q)*yold,0);
                err = norm(ys - yold);
                yold = ys;
                count = count + 1;
               
                Y = MantHelper.circShiftVec(ys,shift)/sqrt(params.N);
                Q = Y'*Y;
%                 Q_hat = D-params.alpha*ones(params.T) - Q;
            end   
        end
        
        % iterative method to solve the self-consistant equations of the
        % NSM
        function Ys = iterationBatchNSM(X,y0,params, step)
%             Yold = toeplitz(y0,y0);
            shift = floor(length(y0)/params.N);
%             Yold = MantHelper.circShiftVec(y0,shift)/sqrt(params.N);
            Yold = MantHelper.circShiftVec(y0,shift);
            D = X'*X;
            maxIter = 1e5;  % maximum number of iterations
            errTol = 1e-6;  % tolarance of error
            err = inf;      % intinial error
            count = 1;

            while(count < maxIter && err > errTol)
%                 Ys = (1-step*params.lambda)*Yold + step*max(Yold*(D -params.alpha*ones(params.T)-Yold'*Yold),0);
                Ys = max(Yold + step*(-params.lambda*Yold + Yold*(D -params.alpha*ones(params.T)-Yold'*Yold)),0);
%                 ys = yold + step*(-params.lambda*yold + max(Q_hat*yold,0));
                err = norm(Ys - Yold)/step;
                Yold = Ys;
                count = count + 1; 
                
                % examing whether the activity explodes
                if max(Ys(:)) > 1e2 || all(Ys(:)==0)
                    disp('Failed to learn, neural activity explodes!')
                    break;
                end
            end   
        end
        
        % iterative method to solve the self-consistant equations of the
        % NSM, batch mode
        function [YEs, YIs] = iterationBatchEI_NSM(X,ye0,yi0, params, step)
            shiftE = floor(length(ye0)/params.NE);
            shiftI = floor(length(yi0)/params.NI);
%             Yold = MantHelper.circShiftVec(y0,shift)/sqrt(params.N);
            Yeold = MantHelper.circShiftVec(ye0,shiftE);
            Yiold = MantHelper.circShiftVec(yi0,shiftI);
%             Yeold = toeplitz(ye0,ye0);
%             Yiold = toeplitz(yi0,yi0);
            D = X'*X;
            maxIter = 1e5;  % maximum number of iterations
            errTol = 1e-6;  % tolarance of error
            err = inf;      % intinial error
            count = 1;

            while(count < maxIter && err > errTol)
                YEs = (1-step*params.lambdaE)*Yeold + step*max(Yeold*(D/params.rhoex + ...
                    Yeold'*Yeold/params.rhoee - Yiold'*Yiold/params.rhoei),0);
                YIs = (1-step*params.lambdaI)*Yiold + step*max(Yiold*(D/params.rhoeix + ...
                    Yeold'*Yeold/params.rhoei - Yiold'*Yiold/params.rhoii),0);
                
                % Use projected gradient-descent
%                 YEs = max(Yeold + step*(- params.lambdaE*Yeold + Yeold*(D/params.rhoex + ...
%                     Yeold'*Yeold/params.rhoee - Yiold'*Yiold/params.rhoei)),0);
%                 YIs = max(Yiold + step*(-params.lambdaI*Yiold+ Yiold*(D/params.rhoex + ...
%                     Yeold'*Yeold/params.rhoei - Yiold'*Yiold/params.rhoii)),0);
                
%                 ys = yold + step*(-params.lambda*yold + max(Q_hat*yold,0));
                err = norm(YEs - Yeold)/step + norm(YIs - Yiold)/step;
                Yeold = YEs;
                Yiold = YIs;
                count = count + 1;
                
                % examing whether the activity explodes
                if max(YEs(:)) > 1e2 || all(YEs(:) < 1e-8) || max(YIs(:)) > 1e2 || all(YIs(:) < 1e-8)
                    disp('Failed to learn, neural activity explodes or all slient!')
                    break;
                end
                
            end   
        end
        
        % iterative method to solve the coupled equations for EI netowrk,
        % only solve ye and yi, not the Gram matrices
        function [YEs, YIs] = iteration_singleEI_NSM(X,ye0,yi0, params, step)
            Yeold = toeplitz(ye0,ye0);
            Yiold = toeplitz(yi0,yi0);
            D = X'*X;
            maxIter = 1e5;  % maximum number of iterations
            errTol = 1e-6;  % tolarance of error
            err = inf;      % intinial error
            count = 1;

            while(count < maxIter && err > errTol)
                YEs = (1-step*params.lambdaE)*Yeold + step*max(Yeold*(D/params.rhoex + ...
                    Yeold'*Yeold/params.rhoee - Yiold'*Yiold/params.rhoei),0);
                YIs = (1-step*params.lambdaI)*Yiold + step*max(Yiold*(D/params.rhoex + ...
                    Yeold'*Yeold/params.rhoei - Yiold'*Yiold/params.rhoii),0);
                
%                 ys = yold + step*(-params.lambda*yold + max(Q_hat*yold,0));
                err = norm(YEs - Yeold) + norm(YIs - Yiold);
                Yeold = YEs;
                Yiold = YIs;
                count = count + 1;
                
            end   
        end
        
        function Y = circShiftVec(x, K)
            % x is a vector 
            % K is an iteger in  the range 1 to length(x) 
            num = floor(length(x)/K);
            Y = nan(length(x),num);
            for i = 1:num
                Y(:,i) = circshift(x(:),(i-1)*K);
            end
            Y = Y';  % make sure Y has the right dimension
        end
        
        % fit  constants
        function [diffConst, alpha] = fitMsd(msd,stepSize,varargin)
        % fit the time dependent msd by anormalous diffusion
        % y = D*x^a, linear regression in the logscale
        % msd is a vector
            if nargin > 2
                time_points = varargin{1}; % number points selected to fit
            else
                time_points = length(msd);
            end

            logY = log(msd(1:time_points));
            logX = [ones(time_points,1),log((1:time_points)'*stepSize)];
            b = logX\logY;  
            diffConst = exp(b(1));  % diffusion constant
            alpha = b(2); % factor
        end
        
        % fit linear diffusion constant for each neuron
        function [Ds,epns] = fitRingDiffusion(msds,stepSize,fitMethod,vargin)
         
         % Only use the linear regime with short time interval
         % msds      mean square displacement array
         % stepSize   dt
         % fitRange   an integer, specifying the
         Ds = nan(size(msds,2),1);
         % linear of log fit
         if strcmp(fitMethod,'linear')
             for i = 1:size(msds,2)
                 % dynamically adjust the fitting range
                 if nargin > 3
                    fitRange = vargin{1};
                 else
                     % use the data that first 2/3 linear regime
                     Len = length(msds(:,i));
                     plateau = mean(msds(round(Len*2/3):end,i));
                     temp = find(msds(:,i) > plateau/2,1,'first');
%                      temp = find(msds(:,i) > 2,1,'first');
                     if isempty(temp)
                         fitRange = 100;
                     else
                         fitRange = temp;
                     end
                 end
                 ys = msds(1:fitRange,i);
                 if ~any(isnan(ys))
                    xs = [ones(fitRange,1),(1:fitRange)'*stepSize];
                    bs = xs\ys;   % linear regression
                    Ds(i) = bs(2);
                 end
             end
         elseif strcmp(fitMethod,'log')
                 
             epns = nan(size(msds,2),1); % store the exponent
             for i = 1:size(msds,2)
                 % dynamically adjust the fitting range
                 if nargin > 3
                    fitRange = vargin{1};
                 else
                     % use the data that first 2/3 linear regime
                     Len = length(msds(:,i));
                     plateau = mean(msds(round(Len*2/3):end,i));
                     temp = find(msds(:,i) > plateau/2,1,'first');
    %                      temp = find(msds(:,i) > 2,1,'first');
                     if isempty(temp)
                         fitRange = 100;
                     else
                         fitRange = temp;
                     end
                 end
                 logY = log(msds(1:fitRange,i));
                 if ~any(isnan(logY))
                     logX = [ones(fitRange,1),log((1:fitRange)'*stepSize)];
                     b = logX\logY;
                     Ds(i) = exp(b(1));
                     epns(i) = b(2);
                 end
             end
         end
         
     end
        
        % single layer neural network decoder
        % simply nonlinear least squre fit
        function wout = fitWout(popVec,xloc)
            fitFun = @(b,x) tanh(b*x);
            beta0 = 0.1*randn(1,size(popVec,1));
            wout = nlinfit(popVec,xloc,fitFun,beta0);
        end
        
        
        % using gradient descent with regularization
        function wout = fitWoutReg(popVec,xloc,learnRate, lbd)
            MaxIter = 1e4;
            ErrTol = 1e-4;
            wout = randn(1,size(popVec,1));
            count = 0;
            err = inf;
            loss = [];
            while count < MaxIter && err > ErrTol
                h = wout*popVec;
                gd = -(tanh(h) - xloc).*(1-tanh(h).^2)*popVec'-lbd*wout;
                loss = [loss, norm(tanh(h)-xloc)^2 + lbd*norm(wout)^2];
                err = norm(learnRate*gd);
                wout = wout + learnRate*gd;
                count = count +1;
            end

        end
        
        % analyze peaks, for Tmaze simulation only
        function peakStats = analyzePksTmaze(Yt,pkThd)
            % return a stuct containing all the statistics of peak info
            % Yt    a k x N x T array of the population response
            % pkThd  threhold used to define if a neuron is silent or not
            
            [k, N, time_points] = size(Yt);
            % peak of receptive field
            peakInxLeft = nan(k,time_points);
            peakValLeft = nan(k,time_points);
            peakInxRight = nan(k,time_points);
            peakValRight = nan(k,time_points);
            for i = 1:time_points
                [pkValL, peakPosiL] = sort(Yt(:,1:round(N/2),i),2,'descend');
                [pkValR, peakPosiR] = sort(Yt(:,(1+round(N/2)):N,i),2,'descend');
                peakInxLeft(:,i) = peakPosiL(:,1);
                peakValLeft(:,i) = pkValL(:,1);
                peakInxRight(:,i) = peakPosiR(:,1);
                peakValRight(:,i) = pkValR(:,1);
            end
            
            % fraction of neurons that are active at any given time
            % quantified by the peak value larger than a threshold 0.01
            rfIndexLeft = peakValLeft > pkThd;
            rfIndexRight = peakValRight > pkThd;
            
            rfIndex = rfIndexLeft | rfIndexRight;

            % fraction of neurons
            activeRatioLeft = sum(rfIndexLeft,1)/k;
            activeRatioRight = sum(rfIndexRight,1)/k;
            activeRatio = sum(rfIndex,1)/k;
            activeRatioBoth = sum(rfIndexLeft & rfIndexRight,1)/k;
            
            peakStats.acRatioL = activeRatioLeft;
            peakStats.acRatioR = activeRatioRight;
            peakStats.actRatio = activeRatio;
            peakStats.actRatioB = peakStats.actRatio;
            
            % select a start index from which the dynamics can be regarded
            % as stationary
            stInx = 101;
            % plot the figure
            figure
            subplot(2,2,1)
            plot(stInx:time_points,activeRatio(stInx:end))
            xlabel('iteration')
            ylabel('active fraction')
            title('Left or Right')
            
            subplot(2,2,2)
            plot(stInx:time_points,activeRatioBoth(stInx:end))
            xlabel('iteration')
            ylabel('active fraction')
            title('Left and Right')
            
            subplot(2,2,3)
            plot(stInx:time_points,activeRatioLeft(stInx:end))
            xlabel('iteration')
            ylabel('active fraction')
            title('Left only')
            
            subplot(2,2,4)
            plot(stInx:time_points,activeRatioRight(stInx:end))
            xlabel('iteration')
            ylabel('active fraction')
            title('Right only')

            % heatmap showing whether a neuron is active or silent ordered
            % by the time point 0
            [~,neuroInxLeft] = sort(rfIndexLeft(:,stInx),'descend');
            [~,neuroInxRight] = sort(rfIndexRight(:,stInx),'ascend');
            
            figure
            subplot(1,2,1)
            imagesc(rfIndexLeft(neuroInxLeft,stInx:end))
            xlabel('iteration')
            ylabel('Ordered neuron index')
            title('Active for Left Trials')
            
            subplot(1,2,2)
            imagesc(rfIndexRight(neuroInxRight,stInx:end))
            xlabel('iteration')
            ylabel('Ordered neuron index')
            title('Active for Right Trials')
            
            % Gain and loss of RF
            sepInx = find(rfIndexLeft(neuroInxLeft,stInx)==0,1);
            leftLoss = 1 - sum(rfIndexLeft(neuroInxLeft(1:(sepInx-1)),stInx:end),1)/(sepInx-1);
            leftGain = sum(rfIndexLeft(neuroInxLeft(sepInx:end),stInx:end),1)/(k-sepInx+1);
            
            sepInxR = find(rfIndexRight(neuroInxRight,stInx)==1,1);
            RightLoss = 1 - sum(rfIndexRight(neuroInxRight(sepInxR:end),stInx:end),1)/(k-sepInxR+1);
            RightGain = sum(rfIndexRight(neuroInxRight(1:(sepInxR-1)),stInx:end),1)/(sepInxR-1);
            
            figure
            subplot(1,2,1)
            plot(leftLoss)
            hold on
            plot(leftGain)
            legend('loss: left','gain:left')
            xlabel('iteration')
            ylabel('fraction')
            
            
            subplot(1,2,2)
            plot(RightLoss)
            hold on
            plot(RightGain)
            legend('loss: right','gain:right')
            xlabel('iteration')
            ylabel('fraction')
            hold off           
            
        end
        
        % gain and loss of peaks, for ring manifold
        function [Loss, Gain] = pksGainLoss(Yt,pkThd)
            % return a stuct containing all the statistics of peak info
            % Yt    a k x N x T array of the population response
            % pkThd  threhold used to define if a neuron is silent or not
            
            [k, N, time_points] = size(Yt);
            % peak of receptive field
            peakInx = nan(k,time_points);
            peakVal = nan(k,time_points);
            for i = 1:time_points
                [pkVal, peakPosi] = sort(Yt(:,:,i),2,'descend');
                peakInx(:,i) = peakPosi(:,1);
                peakVal(:,i) = pkVal(:,1);
            end
            
            % fraction of neurons that are active at any given time
            % quantified by the peak value larger than a threshold 0.01
            rfIndex = peakVal > pkThd;
            

            % fraction of neurons
            activeRatio = sum(rfIndex,1)/k;
            
            peakStats.actRatio = activeRatio;
            peakStats.actRatioB = peakStats.actRatio;

            % heatmap showing whether a neuron is active or silent ordered
            % by the time point 0
            stInx = 101;
            [~,neuroInx] = sort(rfIndex(:,stInx),'descend');
            
            figure
            imagesc(rfIndex(neuroInx,stInx:end))
            xlabel('iteration')
            ylabel('Ordered neuron index')
            title('Active for Left Trials')
            
            % Gain and loss of RF
            sepInx = find(rfIndex(neuroInx,stInx)==0,1);
            Loss = 1 - sum(rfIndex(neuroInx(1:(sepInx-1)),stInx:end),1)/(sepInx-1);
            Gain = sum(rfIndex(neuroInx(sepInx:end),stInx:end),1)/(k-sepInx+1);
 
            figure
            plot(Loss)
            hold on
            plot(Gain)
            legend('loss: left','gain:left')
            xlabel('iteration')
            ylabel('fraction')
  
        end
        
    end
end