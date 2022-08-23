classdef SMhelper < handle
    % this class put all the helper funcitons togehter when analyzing the
    % drift presentations for the PSP task
    
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
     function randEigs = genRandEigs(k,n,rho, distr)
        % generate random eigenvalues, with first k explain fraction of rho
        % variation of the total variation, with different distribution
        if strcmp(distr, 'rand')
            firstK = rand(k,1);
        elseif strcmp(distr, 'lognorm')
            firstK = exp(randn(k,1));
        elseif strcmp(distr, 'exp')
            firstK = exp(rand(k,1));
        end
        
        % fix the summation of eigenvalues to be 10, for convinience
        rest = rand(n-k,1);
        randEigs = [sort(firstK./sum(firstK)*rho,'descend'); rest*(1-rho)./sum(rest)]*10;
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
     
     function angles = eluerAngles(eigVecs)
         % return all the euler anlges of the principle axis
         angles = nan(size(eigVecs,1),size(eigVecs,3)); % store the Euler angles
         refx = [1;0;0];
         refz = [0;0;1];
%          eigVecs = nan(size(Y,1),size(Y,1),size(Y,3));
         for i = 1:size(eigVecs,3)
%              [V,~] = eigs(Y(:,:,i)*Y(:,:,i)');
%              eigVecs(:,:,i) = eigVecs(:);
             V = eigVecs(:,:,i);
             lineNodeVec = cross(refz,V(:,3));
             angles(1,i) = atan2(norm(cross(lineNodeVec,refx)), dot(lineNodeVec,refx));
             angles(2,i) = atan2(norm(cross(lineNodeVec,V(:,1))), dot(lineNodeVec,V(:,1)));
             angles(3,i) = atan2(norm(cross(V(:,3),refz)), dot(V(:,3),refz));
         end
     end

     function msd_tot = rotationalMSD(Yt)
         % define the rotational mean square and fit the diffusion
         % constants
         [k,N,samples] = size(Yt);
         drs = Yt(:,2:end,:) - Yt(:,1:(N-1),:);
         lens = sum(Yt(:,2:end,:).^2,1);      % square of vectors' lengths
         angVel = cross(Yt(:,1:(N-1),:), drs, 1)./lens; % angular velocity
         
         % cumulative angulars
         cumAngs = cumsum(angVel,2);      % cumulated angles
         time_points = size(cumAngs,2);
%          num_sel = size(angles,1);
         msd_tot = nan(samples,floor(time_points/2));
%          msd_comp = nan(samples,floor(time_points/2),size(Yt,1));
         for i = 1:floor(time_points/2)
            diffLag = cumAngs(:,i+1:end,:) - cumAngs(:,1:end-i,:);
            msd_tot(:,i) = squeeze(mean(sum(diffLag.*diffLag,1),2));
         end
         
     end
     
     function [Dphi,exponent] = fitRotationDiff(rmsd, stepSize, fitRange, fitMethod, vargin)
         % Fit the rotational diffusion constant based on mean-squared
         % angular displacement
         if nargin > 4
             plotFlag = vargin{1};   % plot or not, default 0
         else
             plotFlag = 0;
         end
         
        
        allDs = nan(size(rmsd,1),1);
        allExpo = nan(size(rmsd,1),1);
        % linear fit
        if strcmp(fitMethod,'linear')
            for i  = 1:size(rmsd,1)
                temp = rmsd(i,:);
                ys = temp(1:fitRange)';
                xs = [ones(fitRange,1),(1:fitRange)'*stepSize];
                bs = xs\ys;   % linear regression
                allDs(i) = bs(2)/4;   % notice the factor 4 is from the definition
            end
            Dphi = nanmean(allDs);     % population average
            exponent = nanmean(allExpo);

        % fit in the logscale to reduce error
         elseif strcmp(fitMethod,'log')
             
             for i  = 1:size(rmsd,1)
                temp = rmsd(i,:);
                logY = log(temp(1:fitRange)');
                logX = [ones(fitRange,1),log((1:fitRange)'*stepSize)];
                b = logX\logY;  
                allDs(i) = exp(b(1))/4;   % notice the scaling factor 4
                allExpo(i) = b(2);
             end
             Dphi = nanmean(allDs);     % population average
             exponent = nanmean(allExpo);
        % using the mean rmsd and linear fit     
        elseif strcmp(fitMethod, 'mean_linear')
             init_remove = 5; % remove the first 10 time points
             ave_rmsd = mean(rmsd,1);
             ys = ave_rmsd(init_remove+1:fitRange)';
             xs = (init_remove+1:fitRange)'*stepSize;
             b = xs\ys;  
             Dphi = b/4;   % notice the scaling factor 4
             exponent = nan;
             
        elseif strcmp(fitMethod, 'mean_log')
             init_remove = 5; % remove the first 10 time points
             ave_rmsd = mean(rmsd,1);
             logY = log(ave_rmsd(init_remove+1:fitRange)');
             logX = [ones(fitRange-init_remove,1),log((init_remove+1:fitRange)'*stepSize)];
             b = logX\logY;  
             Dphi = exp(b(1))/4;   % notice the scaling factor 4
             exponent = b(2);
        end         
         
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
         end    
     end
     
     % adpative fitting range
     function [Ds,epns] = fitDiffusionAdapt(msds,stepSize,fitMethod,vargin)
         
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
        
        timeLags = min(800,round(0.9*timePoint));
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