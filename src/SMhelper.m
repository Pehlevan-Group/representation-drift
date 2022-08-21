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
     
%      function [diffConst, alphas] = diffConstAngles(angles,stepSize,varargin)
%          % fit a mean square displacement of the angles
%          % angles     a 3 by steps matrix storing the three Euler angles
%          % stepsize   time interval of two adjacent data
%          % fitRange   integer, number of points used to fit msd
%         time_points = size(angles,2);
%         num_sel = size(angles,1);
%         msd = nan(num_sel,floor(time_points/2));
%         for i = 1:floor(time_points/2)
%             diffLag = angles(:,i+1:end) - angles(:,1:end-i);
%             msd(:,i) = mean(diffLag.*diffLag,2);
%         end
%         
%         % fit msd
%         if nargin > 2
%             fitRange = varargin{1}; % number points selected to fit
%         else
%             fitRange = size(msd,2);
%         end
%         % x0 = 1:length(msd)'*stepSize;
%         diffConst = nan(3,1); % diffusion constant of three angles
%         alphas = nan(3,1);  % corresponding exponents
%         for i = 1:3
%             logY = log(msd(i,1:fitRange)');
%             logX = [ones(fitRange,1),log((1:fitRange)'*stepSize)];
%             b = logX\logY;  
%             diffConst(i) = exp(b(1));  % diffusion constant
%             alphas(i) = b(2); % factor
%         end
%         
%         % plot the figure
%         figure
%         plot((1:size(msd,2))'*stepSize,msd')
%         lglables = {['D_{\alpha} = ', num2str(diffConst(1))],['D_{\beta} = ', num2str(diffConst(2))],...
%             ['D_{\gamma} = ', num2str(diffConst(3))]};
%         lg = legend(lglables);
%         set(lg,'FontSize',16)
% %         lg = legend('\alpha','\beta','\gamma','Location','northwest');
% %         set(lg)
%         xlabel('$ \Delta t$','Interpreter','latex','FontSize',28)
%         ylabel('$\langle (\Delta r_{\theta}(t))^2)\rangle$','Interpreter','latex','FontSize',28)
%         set(gca,'FontSize',24,'LineWidth',1.5,'XScale','log','YScale','log')
%         
%         figure
%         plot((1:size(angles,2))'*stepSize,angles')
%         legend('\alpha','\beta','\gamma','Location','northeast');
%         xlabel('$t$','Interpreter','latex','FontSize',28)
%         ylabel('angle (rad)','FontSize',28)
%         set(gca,'FontSize',24,'LineWidth',1.5)
%      end
     
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
         end
         % return the average
         Dphi = nanmean(allDs);     % population average
         exponent = nanmean(allExpo);
         
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
         
%      function plots(Yt, norm_msd,msd_dtheta, step)
%     % plot a gif of 3d scattering of two example samples
% 
%     % first, select two exmaples
%     inxs = randperm(size(Yt,3),2);
%     Y1 = Yt(:,:,inxs(1)); Y2 = Yt(:,:,inxs(2));
% 
%     % define the colors
%     set1 = brewermap(7,'Set1');
%     rdbu = brewermap(11,'RdBu')';
% 
%     figure
%     plot3(Y1(1,:),Y1(2,:),Y1(3,:),'.')
%     hold on
%     plot3(Y2(1,:),Y2(2,:),Y2(3,:),'.')
%     xlabel('$y_1$','Interpreter','latex','FontSize',24)
%     ylabel('$y_2$','Interpreter','latex','FontSize',24)
%     zlabel('$y_3$','Interpreter','latex','FontSize',24)
%     grid on
%     set(gca,'FontSize',20)
%     
%     
%     % encode the temporal information by colors
%     colors = brewermap(size(Y1,2),'Spectral');
%     figure
%     for i = 1:size(Y1,2)
%         plot3(Y1(1,i),Y1(2,i),Y1(3,i),'.','Color',colors(i,:))
%         hold on
%     end
%     xlabel('$y_1$','Interpreter','latex','FontSize',24)
%     ylabel('$y_2$','Interpreter','latex','FontSize',24)
%     zlabel('$y_3$','Interpreter','latex','FontSize',24)
%     hold off
% 
%     % produce a gif and store it
%     az = 45;
%     el = 30;
%     view([az,el])
%     degStep = 5;
%     detlaT = 0.05;
%     % fCount = 71;
%     f = getframe(gcf);
%     [im,map] = rgb2ind(f.cdata,256,'nodither');
%     % im(1,1,1,fCount) = 0;
%     k = 1;
% 
%     for i = 0:degStep:315
%       az = i;
%       view([az,el])
%       f = getframe(gcf);
%       im(:,:,1,k) = rgb2ind(f.cdata,map,'nodither');
%       k = k + 1;
%     end
%     imwrite(im,map,'scatter3D.gif','DelayTime',detlaT,'LoopCount',inf)
% 
% 
%     % linear scale for euclidean
%     X = (1:size(norm_msd,1))'*step;
%     figure
%     for i = 1:5
%         plot(X,norm_msd(:,i),'LineWidth',2,'Color',set1(i,:))
%         hold on
%     end
%     hold off
%     xlabel('$ t$','Interpreter','latex','FontSize',28)
%     ylabel('$\langle (\Delta r(t))^2)\rangle$','Interpreter','latex','FontSize',28)
%     set(gca,'FontSize',24,'LineWidth',1.5)
%     % log scale
%     figure
%     for i = 1:5
%         plot(X,norm_msd(:,i),'LineWidth',2,'Color',set1(i,:))
%         hold on
%     end
%     hold off
%     xlabel('$ t$','Interpreter','latex','FontSize',28)
%     ylabel('$\langle (\Delta r(t))^2)\rangle$','Interpreter','latex','FontSize',28)
%     set(gca,'FontSize',24,'LineWidth',1.5,'XScale','log','YScale','log')
% 
%     X = (1:size(msd_dtheta,1))'*step;
%     figure
%     for i = 1:5
%         plot(X,msd_dtheta(:,i),'LineWidth',2,'Color',set1(i,:))
%         hold on
%     end
%     hold off
%     xlabel('$ t$','Interpreter','latex','FontSize',28)
%     ylabel('$\langle (\Delta r_{\theta}(t))^2)\rangle$','Interpreter','latex','FontSize',28)
%     set(gca,'FontSize',24,'LineWidth',1.5)
%     % log scale
%     figure
%     for i = 1:5
%         plot(X,msd_dtheta(:,i),'LineWidth',2,'Color',set1(i,:))
%         hold on
%     end
%     hold off
%     xlabel('$ t$','Interpreter','latex','FontSize',28)
%     ylabel('$\langle (\Delta r_{\theta}(t))^2)\rangle$','Interpreter','latex','FontSize',28)
%     set(gca,'FontSize',24,'LineWidth',1.5,'XScale','log','YScale','log')
% 
%     % trajectory of the first 3 elements
%     figure
%     plot((1:size(Yt,2))'*step,Yt(:,:,inxs(1))')
%     xlabel('$ t$','Interpreter','latex','FontSize',28)
%     ylabel('$ y_i$','Interpreter','latex','FontSize',28)
%     set(gca,'FontSize',24,'LineWidth',1.5)
% 
%     end

%     function plot3dScatter(Y, vargin)
%     % plot a 3 D scattering of the matrix Y
%     % Y   a 3-d array, num_dimension, num_samples, num_time points
%     [k, N, S] = size(Y);
%    
%     if nargin > 1
%         % do PCA
%     end
% 
%     figure
%     for i = 1:S
%         plot3(Y(1,:,i),Y(2,:,i),Y(3,:,i),'.')
%         hold on
%         xlabel('$y_1$','Interpreter','latex','FontSize',24)
%         ylabel('$y_2$','Interpreter','latex','FontSize',24)
%         zlabel('$y_3$','Interpreter','latex','FontSize',24)
%         grid on
%         set(gca,'FontSize',20)
%     end
%     
%     
%     end

%     function out = hebbianReadout(Y,W,actiFun, vargin)
    % this function test the invariance of hebbian readout
    % Y is the representational matrix
    % W is the readout matrix
    % actiFun is the activation function

    % bias term, should have the same length as the second dim of Y
%     if nargin > 3
%         b = vargin{1};
%     end
%   
%     % initialize the output
%     out = zeros(size(W,1),size(Y,2),size(Y,3));
%     for i = 1:size(Y,3)
%         out(:,:,i) = W*Y(:,:,i);
%     end
%     if strcmp(actiFun,'linear')
%         out = squeeze(out);
%     elseif strcmp(actiFun,'relu')
%         out = squeeze(max(out,0));
%     elseif strcmp(actiFun,'sigmoidal')
%         out = squeeze(1./(1+exp(-out + b)));
%     elseif strcmp(actiFun,'heaviside')
%         out = squeeze(heaviside(out));
% 
%     end
%     end
       
    % regarded the clouds of input as regid body problem
%     function out = rigidBody(Y)
%         % center of mass
%         centerMass = mean(Y,3); % center of mass
%         residue = Y - centerMass;
%         momentIner = sum(sum(residue.^2,3),1);
%         out = [mean(momentIner), std(momentIner)];  % mean and 
%     end
      
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