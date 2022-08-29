classdef PlaceCellhelper < handle
    % this class put all the helper funcitons togehter when analyzing the
    % drift presentations
    
   properties

   end
   
   methods(Static) 

     % generate grid fields for a grid cell
     function gr = gridModule(lbd,theta,r0,ps)
         % return the grid field of a specific module with spacing lbd,
         % rotation theta and offset r0
         % matrix U, 2 x3
         U = [cos(2*pi*(1:3)/3 + theta); sin(2*pi*(1:3)/3 + theta)];
         % R matrix
         
         % grid field for a specific lbd
         gridFields = nan(ps,ps);
         for i = 1:ps
            for j = 1:ps
                r = [i/ps; j/ps];
                R = (r-r0)*ones(1,3);
%                 r0 = [param.x0(1);param.y0(1)];
                gridFields(i,j) = (mean(cos(4*pi/sqrt(3)/lbd*sum(U.*R,1))) + 1/2)*2/3;
            end
         end
         gr = gridFields(:);  % vectorize before sending back
     end
     
     % generate 1D slice of of a 2D grid fields
     function gridFields = gridModuleSlices(lbd,theta,r0,ps,ori)
         % return the grid field of a specific module with spacing lbd,
         % rotation theta and offset r0
         % matrix U, 2 x3
         U = [cos(2*pi*(1:3)/3 + theta); sin(2*pi*(1:3)/3 + theta)];
         % R matrix
         
         % grid field for a specific lbd
         gridFields = nan(ps,1);
         for i = 1:ps
             r = [1;tan(ori)]*i/ps;
             R = (r-r0)*ones(1,3);
             gridFields(i) = (mean(cos(4*pi/sqrt(3)/lbd*sum(U.*R,1))) + 1/2)*2/3;

         end
%          gr = gridFields(:,1);  % vectorize before sending back
     end
     
     % shifted grid field before 1d slicing
     function gridFields = gridModuleSlicesShifted(lbd,theta,r0,ps,ori,dth,dr)
         % return the grid field of a specific module with spacing lbd,
         % rotation theta and offset r0
         % matrix U, 2 x3
         % dth and dr are shifted angle and shifted length to r
         U = [cos(2*pi*(1:3)/3 + theta); sin(2*pi*(1:3)/3 + theta)];
         % R matrix
         
         % grid field for a specific lbd
         gridFields = nan(ps,1);
         for i = 1:ps
             r = [1;tan(ori)]*i/ps + [cos(dth);sin(dth)]*dr;
             R = (r-r0)*ones(1,3);
             gridFields(i) = (mean(cos(4*pi/sqrt(3)/lbd*sum(U.*R,1))) + 1/2)*2/3;
         end
     end
     
     % generate grid fields for a grid cell
     function gr = gridModule1D(lbd,theta,ps)
         % return the grid field of a specific module with spacing lbd,
         %  theta  is the phase
         gr = max(sin((1:ps)/ps/lbd*2*pi + theta),0);
%          gr = abs(sin((1:ps)/ps/lbd*pi + theta));
         gr = gr';  % vectorize before sending back
     end

     % simple non-negative similarity matching
     function states = nsmDynBatch(X,Y, param)
            MaxIter = 1e4;   % maximum iterations
            ErrTol = 1e-4;   % error tolerance
            count = 0;
            err = inf;
            cumErr = inf;
            uyold = zeros(size(Y));
            Yold = Y;
            T = size(X,2);  % number of samples
            errTrack = rand(5,1); % store the lated 5 error
            cumErrTol = 1e-8;
            dt = param.gy;
            
            while count < MaxIter && err > ErrTol && cumErr > cumErrTol
%                 dt = max(param.gy/(1+count/10),1e-2); % using decaying learning rate 
                uy = uyold + dt*(-uyold + param.W*X - sqrt(param.alpha)*param.b - ...
                    (param.M - diag(diag(param.M)))*Yold);
                Y = max((uy - param.lbd1)./(param.lbd2 + diag(param.M)), 0);
                err = norm(Y-Yold,'fro')/(norm(Yold,'fro')+ 1e-10)/dt;  % previously divided by T
                Yold = Y;
                uyold = uy;
                count = count + 1;
                errTrack = [err;errTrack(1:4)];
                cumErr= abs(sum(diff(errTrack)));
            end
            states.Y = Y;
     end
     
     % simple non-negative similarity matching with inhibitory neurons
     function states = nsmDynBatchExciInhi(X,Y,Z,param)
            MaxIter = 1e4;   % maximum iterations
            ErrTol = 1e-4;   % error tolerance
            count = 0;
            err = inf;
            cumErr = inf;
            
            uyold = zeros(size(Y));
            uzold = zeros(size(Z));
            Yold = Y;
            Zold = Z;
            errTrack = rand(5,1); % store the lated 5 error
            cumErrTol = 1e-8;
%             dt = param.gy;
            
            while count < MaxIter && err > ErrTol && cumErr > cumErrTol
%                 Y = max(Y + param.gy*(param.W*X - param.Wei*Zold - sqrt(param.alpha)*param.b), 0);
%                 Z = max(Z + param.gz*(-param.beta*Zold + param.Wie*Yold),0);
                uy = uyold + param.gy*(-uyold + param.W*X - sqrt(param.alpha)*param.b - ...
                    param.Wei*Zold);
                Y = max((uy - param.lbd1)./param.lbd2, 0);
                
                uz = uzold + param.gz*(-uzold + param.Wie*Yold - (param.M - diag(diag(param.M)))*Zold);
                Z = max(uz./diag(param.M),0);
                % try new learning dynamics
%                 Z = max(Zold + param.gz*(-Zold + param.Wie*Yold), 0);
%                 Z = max(Zold + param.gz*(-param.M*Zold + param.Wie*Yold), 0);
                err = norm(Y-Yold,'fro')/(norm(Yold,'fro')+ 1e-10)/param.gy +...
                    norm(Z-Zold,'fro')/(norm(Zold,'fro')+ 1e-10)/param.gz;  % previously divided by T
                Yold = Y;
                Zold = Z;
                count = count + 1;
                errTrack = [err;errTrack(1:4)];
                cumErr= abs(sum(diff(errTrack)));
            end
            states.Y = Y;
            states.Z = Z;
     end
     
     % simple non-negative similarity matching
     function states = sparseCodingDynBatch(X,Y, param)
            MaxIter = 1e4;   % maximum iterations
            ErrTol = 1e-4;   % error tolerance
            count = 0;
            err = inf;
            cumErr = inf;
            uyold = zeros(size(Y));
            Yold = Y;
            T = size(X,2);  % number of samples
            errTrack = rand(5,1); % store the lated 5 error
            cumErrTol = 1e-8;
            dt = param.gy;
            ymax = 10;     % maximum firing rate
            
%             M = param.W*param.W' - eye(param.Np);
            fwd_input = param.W*X;   % forward input
            while count < MaxIter && err > ErrTol && cumErr > cumErrTol
%                 dt = max(param.gy/(1+count/10),1e-2); % using decaying learning rate
                
                uy = uyold + dt*(-uyold + fwd_input - param.M*Yold);
                Y = min(max(uy - param.lbd1, 0),ymax);
                err = norm(Y-Yold,'fro')/(norm(Yold,'fro')+ 1e-10)/dt;  % previously divided by T
                Yold = Y;
                uyold = uy;
                count = count + 1;
                errTrack = [err;errTrack(1:4)];
                cumErr= abs(sum(diff(errTrack)));
            end
            states.Y = Y;
     end
     
     % multiple timescales (multiple weights)
     function states = nsmDynBatchMultiScale(X,Y, param)
            MaxIter = 1e4;   % maximum iterations
            ErrTol = 1e-4;   % error tolerance
            count = 0;
            err = inf;
            cumErr = inf;
            uyold = zeros(size(Y));
            Yold = Y;
%             T = size(X,2);  % number of samples
            errTrack = rand(5,1); % store the lated 5 error
            cumErrTol = 1e-8;
%             dt = param.gy;
            
            h_fwd = (param.W + param.Wslow)*X/2;
            Mhat = (param.M - diag(diag(param.M)) + param.Mslow - diag(diag(param.Mslow)))/2;
            while count < MaxIter && err > ErrTol && cumErr > cumErrTol
                dt = max(param.gy/(1+count/10),1e-2); % using decaying learning rate
                uy = uyold + dt*(-uyold + h_fwd - sqrt(param.alpha)*param.b - Mhat*Yold);
                Y = max((uy - param.lbd1)./(param.lbd2 + diag(param.M)/2 + diag(param.Mslow)/2), 0);
                err = norm(Y-Yold,'fro')/(norm(Yold,'fro')+ 1e-10)/dt;  % previously divided by T
                Yold = Y;
                uyold = uy;
                count = count + 1;
                errTrack = [err;errTrack(1:4)];
                cumErr= abs(sum(diff(errTrack)));
            end
            states.Y = Y;
     end
     
     % bath dynamics for multiple maps
     function states = multipleMapDynBatch(X, param)
            MaxIter = 1e3; % maximum iterations
            ErrTol = 1e-4; % error tolerance
            count = 0;
            err = inf;
            cumErr = inf;
            
            Yold = zeros(param.Np, size(X,2));
            Y = Yold;
            errTrack = rand(5,1); % store the lated 5 error
            cumErrTol = 1e-8;
            
            dt = 0.05;
            % using decaying learning rate
            while count < MaxIter && err > ErrTol && cumErr > cumErrTol
%                 dt = max(params.gy/(1+count/10),1e-2);
                
                Y = Y + dt*(-Y + max(tanh(param.alpha*param.W*X - ones(param.Np,1)*param.J*mean(Y,1) - param.lbd),0));           
                err = norm(Y-Yold,'fro')/(norm(Yold,'fro')+ 1e-10)/dt;  % previously divided by T
                Yold = Y;
                count = count + 1;
                errTrack = [err;errTrack(1:4)];
                cumErr= abs(sum(diff(errTrack)));
            end
            states = Y;
     end
     
     % using quadratic programing to find the fixed point
     % only works when l1 = 0
     function ys = quadprogamYfixed(x,param)
        
        dimOut = size(param.M,1);
        Mbar = (1+param.lbd2)*param.M;
        ys = nan(dimOut,size(x,2));
        for j = 1:size(x,2)
            Q = -(param.W*x(:,j) - param.alpha*param.b);
            ys(:,j) = quadprog(Mbar,Q,-eye(dimOut),zeros(dimOut,1));
        end
    end
     
     % generate next step input from a random walk
     function [ix,iy] = nextPosi(xold,yold,param)
         % return a next step position index, assuming periodic boundary
         % condition
         % x0, y0 are integers in the range of 1 to param.ps
         % param.rwSpeed   steps to take
         for i = 1:param.rwSpeed
             rd = rand;
             if rd <0.5
                 temp = 2*round(rand)-1;
                 if xold + temp > param.ps
                     ix = 1;
                 elseif xold + temp == 0
                     ix = param.ps;
                 else
                     ix = xold + temp;
                 end
                 iy = yold;
                 xold = ix;
                 yold = iy;
             else
                 temp = 2*round(rand)-1;
                 if yold + temp > param.ps
                     iy = 1;
                 elseif yold + temp == 0
                     iy = param.ps;
                 else
                     iy = yold + temp;
                 end
                 ix = xold;
                 
                 xold = ix;
                 yold = iy;
             end
         end
         
     end
     
     % generate next step input from a random walk, 1D
     function xold = nextPosi1D(xold,param)
         % return a next step position index, assuming periodic boundary
         % condition
         % x0, an integer in the range of 1 to param.ps
         % param.rwSpeed   steps to take
         for i = 1:param.rwSpeed
%              rd = rand;
             temp = 2*round(rand)-1;
             if xold + temp > param.ps
                 xold = 1;
             elseif xold + temp == 0
                 xold = param.ps;
             else
                 xold = xold + temp;
             end
         end
         
     end
     
     % offline neural dynamics, with z dynamics
     function [states, params] = neuralDynBatch(X,Y,Z,V, params)
        MaxIter = 1e4; % maximum iterations
        ErrTol = 1e-4; % error tolerance
        count = 0;
        err = inf;
        Yold = Y;
        Zold = Z;
        T = size(X,2);  % number of samples, batch size

        while count < MaxIter && err > ErrTol
            Y = max(Y + params.gy*(params.W*X - V'*Z - sqrt(params.alpha)*params.b), 0);
            Z = max(Z + params.gz*(-params.beta*Z + V*Y),0);
            V = max(V + params.gv*(Z*Y'/T - V),0);
            err = norm(Y-Yold,'fro')/(norm(Yold,'fro')+ 1e-10)/params.gy + norm(Z-Zold,'fro')/(norm(Zold,'fro')+ 1e-10)/params.gy;
            Yold = Y;
            Zold = Z;
            count = count + 1;
        end
        states.Y = Y;
        states.Z = Z;
        states.V = V;
     end
     
     % center of mass of place field 2D
     function [cmxy, pkMass] = centerMassPks(Ys,param, thd)
        % Ys    a array of response to all input
        % thd   the threshold to identify a response to be active
        cmxy = nan(size(Ys,1),2);    % x and y cooridnate
        pkMass = nan(size(Ys,1),1);   % average peak values
        flag = find(sum(Ys > thd,2) > 4);  % 
%         Yt = reshape(Ys,[],param.ps,param.ps);
        
        for i0 = 1:length(flag)
            temp = reshape(Ys(flag(i0),:),param.ps,param.ps);
            [ix, iy, vals] = find(temp);
%             [ix, iy] = find(temp > thd);
%             vals = temp(temp > thd);
            pkMass(flag(i0)) = sum(vals);
            cmxy(flag(i0),:) = mean([ix.*vals,iy.*vals]/mean(vals),1);
        end
     end
    
     % center of mass of place field
     function [cmxy, pkMass] = centerMassPks1D(Ys, thd)
        % Ys    a array of response to all input
        % thd   the threshold to identify a response to be active
        cmxy = nan(size(Ys,1),1);       %cooridnate of center of mass
        pkMass = nan(size(Ys,1),1);     % average peak values
        flag = find(sum(Ys > thd,2) > 3);  % 
%         Yt = reshape(Ys,[],param.ps,param.ps);
        
        for i0 = 1:length(flag)
%             temp = reshape(Ys(flag(i0),:),param.ps);
            [~,ix, vals] = find(Ys(flag(i0),:));
            pkMass(flag(i0)) = sum(vals);
            cmxy(flag(i0)) = mean(ix.*vals/mean(vals));
        end
     end
      
     function diffs = meanSquareDisp(seq,lr_range)
        % return the mean square displacement
        % depending on the dimension of seq, output could be a vector or a matrix
        % seq           data array for peak positions
        % lr_range      the range of data used to do regression
        time_points = size(seq,1);
        num_sel = size(seq,2);
        msd = nan(floor(time_points/2),num_sel);
        for i = 1:floor(time_points/2)
            diffLag = seq(i+1:end,:) - seq(1:end-i,:);
        %     jump = -fix(diffLag/5)*2*pi;
        %     jump = abs(diffLag) > 5;
        %     diffLag(jump) = 0;
        %     diffLagNew = diffLag + jump;
            msd(i,:) = mean(sum(diffLag.*diffLag,1),1);
        %     msd(i,:) = mean(sum(diffLagNew.*diffLagNew,1),1);
        end

        % linear regression to get the diffusion constant

        if lr_range > 1
            x0 = (1:lr_range)';
        else
            x0 = (1:floor(time_points*lr_range))';
        end

        % normalize msd to its radius
        diffs = nan(num_sel,1);
        for i = 1:num_sel
            diffs(i) = x0\msd(1:length(x0),i);
        end

      end
               
     function msd_tot = rotationalMSD(Yt)
         % define the rotational mean square and fit the diffusion
         % constants
         [k,N,samples] = size(Yt);
         drs = Yt(:,2:end,:) - Yt(:,1:(N-1),:);
         lens = sum(Yt(:,2:end,:).^2,1);  % square of vectors
         angVel = cross(Yt(:,1:(N-1),:), drs, 1)./lens; % angular velocity
         
         % cumulative angulars
         cumAngs = cumsum(angVel,2); % cumulated angles
         time_points = size(cumAngs,2);
%          num_sel = size(angles,1);
         msd_tot = nan(samples,floor(time_points/2));
%          msd_comp = nan(samples,floor(time_points/2),size(Yt,1));
         for i = 1:floor(time_points/2)
            diffLag = cumAngs(:,i+1:end,:) - cumAngs(:,1:end-i,:);
            msd_tot(:,i) = squeeze(mean(sum(diffLag.*diffLag,1),2));
         end
         
      end
     
     % fit linear diffusion constant for each neuron
     function [Ds,epns] = fitLinearDiffusion(msds,stepSize,fitMethod)
         % Only use the linear regime with short time interval
         % msds      mean square displacement array
         % stepSize   dt
         % fitRange   an integer, specifying the
         Ds = nan(size(msds,2),1);
         % linear of log fit
         if strcmp(fitMethod,'linear')
             for i = 1:size(msds,2)
                 % specifiy the fitting length
                 Len = length(msds(:,i));
                 plateau = mean(msds(round(Len/2):end,i));
                 temp = find(msds(:,i) > plateau/3,1,'first');
    %                      temp = find(msds(:,i) > 2,1,'first');
                 if isempty(temp)
                     assert(Len > 100,'trajectory length is not long enough!')
                     fitRange = 100;
                 else
                     fitRange = temp;
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
                 
                 Len = length(msds(:,i));
                 plateau = mean(msds(round(Len/2):end,i));
                 temp = find(msds(:,i) > plateau/2,1,'first');
    %                      temp = find(msds(:,i) > 2,1,'first');
                 if isempty(temp)
                     fitRange = 100;
                 else
                     fitRange = temp;
                 end
                 
                 if sum(~isnan(msds(1:fitRange,i)))>10
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
         
      end
     
     function [Dphi,exponent] = fitRotationDiff(rmsd, stepSize, fitRange, vargin)
         % plot and return the diffusion constant of rotationl angles
         if nargin > 3
             plotFlag = vargin{1};   % plot or not, default 0
         else
             plotFlag = 0;
         end
         % fit the average rotational diffusion constants
         aveRMSD = mean(rmsd,1);
         logY = log(aveRMSD(1:fitRange)');
         logX = [ones(fitRange,1),log((1:fitRange)'*stepSize)];
         b = logX\logY;  
         Dphi = exp(b(1));  % diffusion constant
         exponent = b(2); % factor
         
         % plot and examing the fit
         if plotFlag
             grs = brewermap(11,'Greys');
             blues = brewermap(11,'Blues');
             figure
             pInx = randperm(size(rmsd,1),10);
             plot((1:size(rmsd,2))'*stepSize, rmsd(pInx,:)','Color',grs(5,:),'LineWidth',2)
             hold on
             plot((1:size(rmsd,2))*stepSize,mean(rmsd,1),'LineWidth',4,'Color',blues(8,:))
             % overlap fitted line
             yFit = exp(logX*b);
             plot((1:fitRange)'*stepSize,yFit,'k--','LineWidth',2)
             hold off
             xlabel('$\Delta t$','Interpreter','latex','FontSize',28)
             ylabel('$\langle\varphi^2(\Delta t)\rangle$','Interpreter','latex','FontSize',28)
             set(gca,'FontSize',24,'LineWidth',1.5,'XScale','log','YScale','log',...
                 'XTick',10.^(1:5),'YTick',10.^(-4:2:0))
             set(gca,'FontSize',24,'LineWidth',1.5)
         end
         
         
      end
     
     function V = newEigVec(new,old)
         % change the sigin to match the previous eigen vectors
         V = new;
         for i = 1:size(new,2)
             if new(:,i)'*old(:,i) < 0
                V(:,i) = -1*V(:,i);
             end
         end
      end

     function [diffConst, alpha] = fitMsd(msd,stepSize,varargin)
        % fit the time dependent msd by anormalous diffusion
        % y = D*x^a, linear regression in the logscale
        % msd is a vector
        if nargin > 2
            time_points = varargin{1}; % number points selected to fit
        else
            time_points = length(msd);
        end
        % x0 = 1:length(msd)'*stepSize;

        logY = log(msd(1:time_points));
        logX = [ones(time_points,1),log((1:time_points)'*stepSize)];
        b = logX\logY;  
        diffConst = exp(b(1));  % diffusion constant
        alpha = b(2); % factor
      end
         
     % auto correlation funcitons, fit exponential
     function [acoef,meanAcf,allTau] = fitAucFun(Y,step)
        % fit individual components
        [k, timePoint, num] = size(Y);
        
        timeLags = 500;
        acoef = nan(timeLags+1,3,num);
        allTau = nan(k,num);  % store all the timescale
        xFit = (1:(timeLags +1))'*step;
        modelfun = @(b,x)(exp(-b*x));  % define the exponential function to fit
        opts = statset('nlinfit');
        opts.RobustWgtFun = 'bisquare';
        for i = 1:num
            for j = 1:k
               yFit = autocorr(squeeze(Y(j,:,i)),'NumLags',timeLags);
               acoef(:,j,i) = yFit;
               allTau(j,i) = nlinfit(xFit,yFit',modelfun,1/(timePoint*step),opts);
            end
        end
        
        % mean time scale
        meanAcf = squeeze(mean(mean(acoef,3),2));

    end
    
   end
end