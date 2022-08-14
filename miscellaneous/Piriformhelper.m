classdef Piriformhelper < handle
    % this class put all the helper funcitons togehter when analyzing the
    % drift presentations in piriform cortex
    
   properties
%       lr
%       t      
%       k
%       d
%       tau
%       Minv
%       W
%       outer_W
%       outer_Minv
%       y
      
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
     
     % generate weakly-tuned MEC cells
     function gridFields = weakMEC(ps, ng, sig)
         % ps linear dimension, an integer
         % ng   number of grid cells
         % sig  standard deviation of smooth kernel
         gridFields = nan(ps*ps,ng);
         for i = 1:ng
             Iblur = imgaussfilt(rand(ps,ps),sig);
             normVec = (Iblur - min(Iblur(:)))/(max(Iblur(:)) - min(Iblur(:)));
             gridFields(:,i) = normVec(:);
             
         end
     end
     % simple non-negative similarity matching
     function states = nsmDynBatch(X,Y, param)
            MaxIter = 1e4; % maximum iterations
            ErrTol = 1e-4; % error tolerance
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
                dt = param.gy;
                uy = uyold + dt*(-uyold + param.W*X - sqrt(param.alpha)*param.b - ...
                    (param.M - diag(diag(param.M)))*Yold);
                Y = max((uy - param.lbd1)./(param.lbd2 + diag(param.M)), 0);
%                 Y = min(1,Y);  % 9/29/2020
                err = norm(Y-Yold,'fro')/(norm(Yold,'fro')+ 1e-10)/T/dt;
                Yold = Y;
                uyold = uy;
                count = count + 1;
                errTrack = [err;errTrack(1:4)];
                cumErr= abs(sum(diff(errTrack)));
            end
            states.Y = Y;
     end
     
     % offline neural dynamics
    function [states, params] = neuralDynBatch(X,Y,Z,V, params)
        MaxIter = 1e3; % maximum iterations
        ErrTol = 1e-4; % error tolerance
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
     
     % place field peak position
     function pkPosi = placePeakPosi(Y, params)
         % Y is the response of all the 
     end
     function diffs = meanSquareDisp(seq,lr_range)
        % return the mean square displacement
        % depending on the dimension of seq, output could be a vector or a matrix
        % seq           data array for
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
         
         % decomposition of the roational angles
%          angProj = nan(size(angVel));
%          for i = 1:size(angVel,2)
%              for j = 1:samples
%                 angProj(:,i,j) = eigVec(:,:,i)'*angVel(:,i,j);
%              end
%          end
         % cumulative angulars
%         cumAngs_comp = cumsum(angProj,2); % cumulated angles
%          time_points = size(cumAngs_comp,2);
%          msd_comp = nan(k,samples,floor(time_points/2));
%          for i = 1:floor(time_points/2)
%             diffLag = cumAngs_comp(:,i+1:end,:) - cumAngs_comp(:,1:end-i,:);
%             msd_comp(:,:,i) = squeeze(mean(diffLag.*diffLag,2));
%          end
%          
%          xlabel('$t$','Interpreter','latex','FontSize',28)
%          ylabel('angle (rad)','FontSize',28)
%          set(gca,'FontSize',24,'LineWidth',1.5)
         
     end
     
     function [Dphi,exponent] = fitRotationDiff(rmsd, stepSize, fitRange, vargin)
         % plot and return the diffusion constant of rotationl angles
         if nargin > 3
             plotFlag = 1;   % plot or not, default 0
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
         
         % plot and save
         if plotFlag
             fFolder = './figures';
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
%              figPref = 'rmsd_same_eig';
%              saveas(gcf,[fFolder,filesep,figPref,'.fig'])
%              print('-depsc',[fFolder,filesep,figPref,'.eps'])
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
         
    function plot3dScatter(Y, vargin)
    % plot a 3 D scattering of the matrix Y
    % Y   a 3-d array, num_dimension, num_samples, num_time points
    [k, N, S] = size(Y);
   
    if nargin > 1
        % do PCA
    end

    figure
    for i = 1:S
        plot3(Y(1,:,i),Y(2,:,i),Y(3,:,i),'.')
        hold on
        xlabel('$y_1$','Interpreter','latex','FontSize',24)
        ylabel('$y_2$','Interpreter','latex','FontSize',24)
        zlabel('$y_3$','Interpreter','latex','FontSize',24)
        grid on
        set(gca,'FontSize',20)
    end
    
    
    end
      
    % auto correlation funcitons, fit exponential
    function [acoef,meanAcf,allTau] = fitAucFun(Y,step)
        % fit individual components
        [k, timePoint, num] = size(Y);
        
        timeLags = 500;
        acoef = nan(timeLags+1,3,num);
        allTau = nan(k,num);  % store all the timescale
%         refY = Yt(:,:,100);  % reference
%         vecRef = refY(:);
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
%         meanTau = fit(xFit,meanAcf,'exp1');
        % fit an exponential decaying curve
        
%         yFit = acfSM(xFit);
%         fexp1 = fit(xFit,yFit,'exp1');
%         fexp2 = fit(xFit,acfCoef(xFit),'exp1');  
        
%         beta2 = nlinfit(xFit,acfCoef(xFit),modelfun,[0,1,1e-2],opts);

    end
    
   end
end