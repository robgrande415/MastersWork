%============================= onlineGP_PE ================================
%
%  This class implements onlineGP with hyperparameter learning. It is a
%  subclass of the onlineGP class. 
%
%  Reference(s): 
%    Sparse Online Gaussian Processes -Csato and Opper, Tech Report
%    Csato's thesis
%    (Add Rob reference)
% 
%  Inputs:
%    sigma  	 - bandwidth for the Gaussian kernel; either
%                  1 x 1 scalar or
%                  1 x d vector
%    noise      -  parameter for noise in GP; assumed given by oracle
%    ncent      -  the size of your budget
%    tol        -  tolerance for projection residual
%
%  Outputs:
%    see functions
%
%============================= onlineGP_PE ================================
%
%  Name:		onlineGP_PE.m
%
%  Author: 		Hassan A. Kingravi
%               Rob Grande (original)
%
%  Created:  	2013/08/12
%  Modified: 	2013/08/12
%
%============================= onlineGP_PE ================================
classdef onlineGP_PE < handle & onlineGP.onlineGP
  % class properties   
  properties (Access = public)
    full           = [];   % variable that checks if budget is hit
    thetaVariance  = 1;    % controls rate of parameter convergence
    thetaEvolution = 1e-5; % controls steady state thetaVariance
    learningRate   = 0.001; % controls rate of changing parameter
    sigmaEstimate  = [];   % used for parameter learning
    noiseEstimate  = [];
    counter        = 1;    % used for parameter convergence    
  end
  
  methods
    %---------------------------- onlineGP_PE -----------------------------
    %
    %  Constructor for onlineGP_PE. 
    %
    %  Inputs:
    %    sigma     - 1 x 1 positive scalar for bandwidth of SE kernel
    %    noise     - 1 x 1 positive scalar for noise of SE kernel
    %    ncent     - 1 x 1 positive scalar for number of centers (budget)
    %    tol       - 1 x 1 positive scalar for tolerance for kernel linear 
    %                      independence test
    %    thV       - 1 x 1 scalar for theta variance 
    %    lR        - 1 x 1 scalar for learning rate
    %    thE       - 1 x 1 scalar for theta evolution
    %
    %  Outputs:
    %    -none
    %(    
    function obj = onlineGP_PE(sigma,noise,ncent,tol,thV,lR,thE)
      % this part initializes the onlineGP superclass
     super_args{1} = [];
     super_args{2} = [];
     super_args{3} = [];
     super_args{4} = [];     
     
      if nargin == 1
        super_args{1} = sigma; 
      elseif nargin == 2
        super_args{1} = sigma; 
        super_args{2} = noise; 
      elseif nargin == 3
        super_args{1} = sigma; 
        super_args{2} = noise; 
        super_args{3} = ncent; 
      elseif nargin >= 4
        super_args{1} = sigma; 
        super_args{2} = noise; 
        super_args{3} = ncent; 
        super_args{4} = tol;        
      end
      
      % now call superclass constructor 
      obj = obj@onlineGP.onlineGP(super_args{:});
                    
      if nargin == 5
        obj.thetaVariance = thV;
      elseif nargin == 6
        obj.thetaVariance = thV;
        obj.learningRate = lR;
      elseif nargin == 7
        obj.thetaEvolution = thE;
      end
      
      % initialize estimates
      obj.sigmaEstimate = obj.sigma;
      obj.noiseEstimate = obj.noise;
      
      % set computation type in updates
      obj.set('reg_type','normal');
    end  
    
    %------------------------------- predict_gradient -------------------------------
    %
    %  Given a new datapoint, predict gradient of GP and second deriv
    %
    %(
    function [g,h] = predict_gradient(obj,x,dim)
      
      k = kernel(x,obj.BV,sigma)';
      g = -((repmat(x(dim,:),1,size(obj.BV,2))-obj.BV(dim,:)).* k')*obj.alpha /obj.sigma^2;
      h = -(  (((repmat(x(dim,:),1,size(obj.BV,2))-obj.BV(dim,:)).^2)...
             /obj.sigma^4 -1/obj.sigma^2) .* k')*obj.alpha /obj.sigma^2;
      
    end
    
    %------------------------------- update_bandwidth --------------------------------
    %
    %  change bandwidth
    %(
    function update_param(obj,x,y)
      
      %optimizing hyperparameters by gradient ascent using log likelihood
      xtrain = [obj.BV, x];
      ytrain = [obj.obs; y];
      
      %for sigma scalar
      if numel(obj.sigma) == 1
        
        [Suu,dSuuDs] = obj.kernel_deriv(xtrain,xtrain,obj.sigma,obj.noise,'sigma');
        %Suu = onlineGP.kernel_deriv(xtrain,xtrain,sigma) + eye(size(xtrain,2))*noise; %using GP
        
        while cond(Suu) > 1e5
          Suu = Suu + obj.noise*eye(size(Suu));
          obj.noise = 2*obj.noise;
        end
        
        iSuu = Suu^-1; % ACTUAL INVERSE
        
        %derivatives
        %[~,dSuuDn] = onlineGP.kernel_deriv(xtrain,xtrain,sigma,noise,'snr'); %identity
        dSuuDn = 2*obj.noise*eye(size(xtrain,2));
        
        %dLDs = 1/2* (ytrain' * iSuu * dSuuDs * iSuu * ytrain  ...
        %    -trace(iSuu * dSuuDs));
        %dLDn = 1/2* (ytrain' * iSuu * dSuuDn * iSuu * ytrain  ...
        %    -trace(iSuu * dSuuDn));
        
        % alternative
        al = iSuu*ytrain;
        dLDs = 1/2*trace( (al*al' - iSuu) * dSuuDs);
        dLDn = 1/2*trace( (al*al' - iSuu) * dSuuDn);
        dLdir = [dLDs, dLDn];
        
        %try convert to likelihood, not log likelihood
        %dLdir = (2*pi*sigma^2)^(-length(ytrain)/2) * exp(-1/2* ytrain'*iSuu*ytrain) .*dLdir
        
        %makes sure derivative doesn't blow up, huge aspect ratio problems
        %with noise derivative being way too big...
        
        %impose max growth rate
        if obj.learningRate*obj.thetaVariance*abs(dLdir(1)) > abs(obj.sigmaEstimate)*0.25
          obj.sigmaEstimate = obj.sigmaEstimate*(1+0.1*sign(dLdir(1)));
        else
          obj.sigmaEstimate = obj.sigmaEstimate + obj.learningRate*obj.thetaVariance*dLdir(1);
        end
        
        if obj.learningRate*obj.thetaVariance*abs(dLdir(2)) > abs(obj.noiseEstimate)*0.25
          obj.noiseEstimate = obj.noiseEstimate*(1+0.1*sign(dLdir(2)));
        else
          obj.noiseEstimate = obj.noiseEstimate + obj.learningRate*obj.thetaVariance*dLdir(2);
        end
        
        %for PSD
        if obj.noiseEstimate <= 1e-3
          obj.noiseEstimate = 1e-3;
        end
        
        %{
            %for conditioning of Suu^-1
            
            if noise > obj.sigmaEstimate/2
                noise = obj.sigmaEstimate/2;
            end
            
            Suu= onlineGP.kernel_deriv(xtrain,xtrain,sigma,noise);
            while cond(Suu) > 1e3
                Suu = Suu + noise*eye(size(Suu));
                noise = 2*noise;
            end
        %}
        %obj.thetaVariance = (1-10e-3)^2 *obj.thetaVariance + thetaEvolution;
        obj.thetaVariance = obj.counter/(obj.counter+1)*obj.thetaVariance;
        obj.counter = obj.counter+1;
        %obj.thetaVariance = obj.learningRate*obj.thetaVariance*dLdir(1:length(sigma));
        
      else
        Suu = onlineGP.kernel_deriv(xtrain,xtrain,obj.sigma) + eye(size(xtrain,2))*obj.noise; %using GP
        iSuu = Suu^-1; %ACTUAL INVERSE
        
        %derivatives
        [~,dSuuDs] = onlineGP.kernel_deriv(xtrain,xtrain,obj.sigma,obj.noise,'sigma');
        %[~,dSuuDn] = onlineGP.kernel_deriv(xtrain,xtrain,obj.sigma,obj.noise,'snr'); %identity
        dSuuDn = eye(size(xtrain,2));
        for i = 1:length(obj.sigma)
          
          dLDs(i) = 1/2* (ytrain' * iSuu * dSuuDs(:,:,i) * iSuu * ytrain  ...
            -trace(iSuu * dSuuDs(:,:,i)));
        end
        dLDn = 1/2* (ytrain' * iSuu * dSuuDn * iSuu * ytrain  ...
          -trace(iSuu * dSuuDn));
        
        dLdir = [dLDs, dLDn];
        %makes sure derivative doesn't blow up, huge aspect ratio problems
        %with noise derivative being way too big...
        for ii = 1:length(dLdir)
          if abs(dLdir(ii)) > 5
            %dLdir(ii) = 5*sign(dLdir(ii));
          end
        end
        obj.sigmaEstimate = obj.sigmaEstimate + obj.learningRate*obj.thetaVariance*dLdir(1:length(obj.sigma));
        obj.noise = obj.noise + 0*obj.learningRate*obj.thetaVariance*dLdir(2); % edit: no noise adjustment
        if obj.noise <= 1e-5
          obj.noise = 1e-5;
        end
        
        %obj.thetaVariance = (1-5e-3)^2 *obj.thetaVariance + thetaEvolution;
        obj.thetaVariance = obj.counter/(obj.counter+1)*obj.thetaVariance;
        obj.counter = obj.counter+1;
      end
      
      % only change if over n% change
      if abs(norm(obj.sigmaEstimate-obj.sigma))/norm(obj.sigma) > 0.05 || ...
          abs(norm(obj.noiseEstimate-obj.noise))/norm(obj.noise) > 0.05
        old_sigma = obj.sigma;
        obj.sigma = obj.sigmaEstimate;
        obj.noise = obj.noiseEstimate;
        %EDIT: BY ROB WEIGHTS NEED TO BE ADJUSTED IF KT IS ADJUSTED
        %update alpha by minimizing LS
        k_t  = onlineGP.kernel(obj.BV,obj.BV,old_sigma,0);
        %k_t1 = obj.kernel_deriv(obj.BV,obj.BV,obj.sigma);
        %old = k_t * alpha;
                
        %as alpha increases or decreases, value added and basis vectors
        %may need to be pruned or added to prevent numerical
        %invertibility problems and maintain coverage of area of interest, respectively.
        
        %get distance between points
        BV_old = obj.BV;
        ind = 1;
        
        %if two points are too close, one gets deleted
        while isempty(ind) == 0
          val = onlineGP.kernel(obj.BV,obj.BV,obj.sigma,0);
                    
          val(eye(size(val))~=0)=0; %ensures the same point isnt selected
          [v, ind] = max(val); % get points closest to each point
          threshold = 0.98;
          v = v - threshold;
          ind = find(v > 0,1);
          obj.BV(:,ind) = [];
          obj.obs(ind) = [];
          if isempty(ind) == 0
            obj.current_size = obj.current_size-1;
          end
        end
        
        jitter = 1e-5;
        k_t1 = onlineGP.kernel(BV_old,obj.BV,obj.sigma,jitter);
        %alphaOld = alpha;
        %reinit alpha by using last information
        
        obj.alpha = k_t1\k_t*obj.alpha;
        %new = k_t1 * alpha;
        obj.K = onlineGP.kernel(obj.BV,obj.BV,obj.sigma,0) + 0*eye(size(obj.BV,2));
        Ktilde = obj.K + obj.noise*eye(size(obj.K));
        %obj.C = -Ktilde^-1;
        %obj.K(1,1) = obj.K(1,1) + obj.noise;
        I = eye(obj.current_size);

        obj.Q = (obj.K + obj.jitter.*eye(obj.current_size))\I; % inverted gram matrix
        obj.C = (obj.K + obj.noise*eye(obj.current_size))\I;
        obj.alpha = -obj.C*obj.obs;
                
        %alternate method of translating covariance matrix
        %C = 1.01*C + eye(size(C))*0.0;
        %C = k_t1^-1 + k_t1^-1*C*k_t1^-1;
        %H = k_t1 * (onlineGP.kernel_deriv(BV,BV,obj.sigma))^-1;
        %cov matrix
        %Kxnew = onlineGP.kernel_deriv(BV,BV,obj.sigma);
        %Kxold = onlineGP.kernel_deriv(BV,BV_old,old_sigma);
        %Kxpold = onlineGP.kernel_deriv(BV_old,BV,old_sigma);
        %Kxpnew = onlineGP.kernel_deriv(BV,BV,obj.sigma);
        %Cnew = Kxnew^-1 * Kxold * C * Kxpold * Kxpnew^-1;
        %C = 0.9999*Cnew;
      end
            
    end
    
    
    

    function [v, d] =  kernel_deriv(obj,x,y,sigma,noise,derivArg)


          if(length(sigma) == 1) %same sigma
                    d=x'*y;
                    dx = sum(x.^2,1);
                    dy = sum(y.^2,1);
                    val = repmat(dx',1,length(dy)) + repmat(dy,length(dx),1) - 2*d;
                    v = exp(-val./(2*sigma^2));
                else
                    isigma = inv(diag(sigma.^2));
                    d =  (x'*isigma)*y;
                    dx = sum((x'*isigma)'.*x,1);
                    dy = sum((y'*isigma)'.*y,1);
                    val = repmat(dx',1,length(dy)) + repmat(dy,length(dx),1) - 2*d;
                    v = exp(-val./2);
          end
          if nargin > 3
              if size(x) == size(y)
                if x==y
                    v = v + eye(size(v))*noise^2;
                end
              end
          end


          if nargout > 1 
            if nargin >=5

              switch (derivArg)
                case{'Amplitude','A','amplitude'}
                    d = exp(-val./(2*sigma^2));
                case{'Sigma','sigma','Bandwidth','S'}
                    if numel(sigma) > 1
                        clear d
                        for i = 1:length(sigma)
                            dist=x(i,:)'*y(i,:);
                            dx = sum(x(i,:).^2,1);
                            dy = sum(y(i,:).^2,1);
                            val2 = repmat(dx',1,length(dy)) + repmat(dy,length(dx),1) - 2*dist;
                            d(:,:,i) = exp(-val./2) .*val2 /sigma(i)^3;
                        end
                    else
                        d = val.*exp(-val./(2*sigma^2)) / sigma^3;
                    end
                case{'noise','snr'}
                    if size(x) == size(y)
                        if x==y
                            d = 2*noise*eye(size(x,2));
                        else
                            d = zeros(size(x,2));
                        end  
                    else
                        d = zeros(size(x,2),size(y,2));
                    end

                end

            else
                fprintf('No derivative argument for kernel \n')
            end
          end
      end

    
    
    
    
    
    
    
    
    
  end
  
end

