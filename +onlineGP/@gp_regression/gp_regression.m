%========================== gp_regression =================================
%  
%  This class implements some basic Gaussian process regression code, which
%  sort of rewraps Rob Grande's code into one object. 
%
%  Reference(s): 
%    C. Rasmussen and C.K Williams 
%       - Gaussian Processes for Machine Learning, Chapter 2.2, pg 19.
%       - Gaussian Processes for Machine Learning, Chapter 2.2, pg 19.
% 
%  Inputs:
%    data	      - n x d data matrix, with each row as a value. 
%    observations - n x 1 vector with each row as an observation
%    sigma  	  - the bandwidth for the kernel; can be a covariance 
%                   matrix or just a scalar parameter
%    noise_p      - estimated noise in observations; must be scalar 
%  Outputs:
%    gpr          - class pointer
%
%========================== gp_regression =================================
%  Name:	gp_regression.m
%
%  Author: 	Hassan A. Kingravi
%
%  Created:  2013/06/26
%  Modified: 2013/08/10
%
%========================== gp_regression =================================
classdef gp_regression < handle 
  % class properties     
  properties (Access = public)
    % default values for constructor: the kernel does not include .A 
    sigma        = 1;
    noise        = 0.1;
    A            = 1; 
  end
  
  % hidden variables 
  properties (Access = protected)
    data     = [];   % data needed for prediction
    obs      = [];   % retained observations
    nsamp    = [];   % number of samples for training data
    K        = [];   % inverse of the kernel matrix
    mean_vec = [];   % mean vector 
  end  
  
  % class methods 
  methods
    
    %--------------------------- gp_regression ----------------------------
    %
    %  Default constructor for gp_regression. 
    %
    %  Inputs:
    %    sigma     - 1 x 1 positive scalar for bandwidth of SE kernel
    %    noise     - 1 x 1 positive scalar for bandwidth of SE kernel
    %    params    - struct for passing into kernel function
    %
    %(    
    function obj = gp_regression(sigma,noise,A)
      try 
      % nesting structure to set new values 
      if nargin >=1
        
        if sigma <= 0
          exception = MException('VerifyOutput:OutOfBounds', ...
                       'Bandwidth must be strictly positive');
          throw(exception);
        end
        
        obj.sigma = sigma;         
        
        if nargin >=2          
          if noise < 0
            exception = MException('VerifyOutput:OutOfBounds', ...
                         'Noise must be positive');
            throw(exception);
          end
          
          obj.noise = noise; 
          
          if nargin >=3
            
            if A <= 0
              exception = MException('VerifyOutput:OutOfBounds', ...
                            'parms.A must be positive');
              throw(exception);
            end
            
            obj.A = A;
                          
          end          
        end  
      end

      catch ME
        disp([ME.message '!'])  
        err = MException('VerifyOutput:OutOfBounds', ...
                'Exiting due to incorrect inputs.');
        throw(err);      
      end
           
    end  
    
    %)
    %------------------------------- train -------------------------------       
    %
    %  Given the data, this function computes all the parameters associated
    %  to the GP model.
    %
    %  Inputs:
    %    training_data - d x nsamp data matrix passed in columnwise
    %    observations  - 1 x nsamp column vector of observations
    %
    %  Outputs:
    %    - none
    %(
    function train(obj,training_data,observations)
      % store in variables
      obj.data = training_data;
      obj.obs  = observations;
      
      obj.nsamp = size(obj.data,2);
      
      % First compute the associated kernel matrix and coefficients.
      % To make the kernel matrix objects more readable, we denote
      % the data in the subscripts
      obj.K = onlineGP.kernel(obj.data,obj.data,obj.sigma,obj.A) + (obj.noise^2)*eye(obj.nsamp);      
      
      % see pg. 19 of Rasmussen for algorithm
      obj.mean_vec = obj.K\transpose(obj.obs);
    end
    
    %)
    %------------------------------- predict -------------------------------
    %
    %  Given new data, predict new values based on current model, and
    %  compute the predictive variance.
    %
    %  Inputs:
    %    te_data      - d x ntest data points passed in columnwise
    %
    %  Outputs:
    %    f  	        - 1 x ntest function value predictions
    %    var_x        - 1 x ntest predictive variances
    %(
    function [f,var,Var] = predict(obj,te_data)
      
      % compute kernel vector
      K_tr_te = onlineGP.kernel(obj.data,te_data,obj.sigma,obj.A);
      
      % compute prediction
      f = transpose(obj.mean_vec)*K_tr_te;
      
      % if necessary, compute predictive variance
      if nargout > 1
        % see pg. 19 of Rasmussen
        K_te_te = onlineGP.kernel(te_data,te_data,obj.sigma,obj.A);
        
        Var = K_te_te - transpose(K_tr_te)*(obj.K\K_tr_te);
        var = transpose(diag(Var));
      end
      
    end
    
    %)
    %----------------------------- log_likelihood ---------------------------
    %
    %  Helper function: compute the log marginal likelihood for an arbitrary
    %  set of data.
    %
    %  Inputs:
    %   new_data      - d x n data points passed in columnwise
    %   new_obs       - 2 x n observations passed in columnwise
    %
    %  Outputs:
    %    likelihood   - 1 x 1 log marginal likelihood
    %(
    function [likelihood] = log_likelihood(obj,new_data,new_obs)
      ntest = size(new_data,2);
      
      % compute kernel matrix
      K_new = onlineGP.kernel(new_data,new_data,obj.sigma,obj.A) + (obj.noise^2).*eye(ntest);
      
      %numerical issues
      counter = 1;
      K_new_temp= K_new;
      while det(K_new_temp)==0
          K_new_temp = K_new_temp*2;
          counter = counter+1;
      end
      temp = new_obs*(K_new\transpose(new_obs)) + log(det(K_new_temp)) - ntest*log(2*counter);
      likelihood = -0.5*temp;
    end
    
    %----------------------------- log_likelihood ---------------------------
    %
    %  Helper function: compute the log marginal likelihood for an arbitrary
    %  set of data.
    %
    %  Inputs:
    %   new_data      - d x n data points passed in columnwise
    %   new_obs       - 2 x n observations passed in columnwise
    %
    %  Outputs:
    %    likelihood   - 1 x 1 log marginal likelihood
    %(
    function [likelihood] = log_likelihood_no_noise(obj,new_data,new_obs)
      ntest = size(new_data,2);
      
      % compute kernel matrix
      K_new = onlineGP.kernel(new_data,new_data,obj.sigma,obj.A);
      
      %numerical issues
      counter = 1;
      K_new_temp= K_new;
      while det(K_new_temp)==0
          K_new_temp = K_new_temp*2;
          counter = counter+1;
      end
      temp = new_obs*(K_new\transpose(new_obs)) + log(det(K_new_temp)) - ntest*log(2*counter);
      likelihood = -0.5*temp;
    end
        
    %)
    %----------------------------- model_prob ----------------------------
    %
    %  Helper function: compute the probability that given data comes from a
    %  given modl.
    %
    %  Inputs:
    %   new_data      - d x n data points passed in columnwise
    %   new_obs       - 2 x n observations passed in columnwise
    %
    %  Outputs:
    %    likelihood   - 1 x 1 log marginal likelihood
    %(
    function [prob] = model_prob(obj,new_data,new_obs)
      ntest = size(new_data,2);
      likelihood = log_likelihood(obj,new_data,new_obs);
      prob = likelihood - ntest/2*log(2*pi);
    end
    
    %)
    %---------------------------- model_prob_covar --------------------------
    %
    %  Helper function: compute the probability that given data comes from a
    %  given GP model when the mean and covariance is available.
    %
    %  Inputs:
    %   new_data      - d x n data points passed in columnwise
    %   new_obs       - 2 x n observations passed in columnwise
    %
    %  Outputs:
    %    likelihood   - 1 x 1 log marginal likelihood
    %(
    function [prob] = model_prob_covar(obj,covar,obs_diff)
      ntest = size(obs_diff,2);
      covar = covar + (obj.noise^2).*eye(ntest);
      
      %numerical issues
      counter = 1;
      covar_temp = covar;
      while det(covar_temp)==0
          covar_temp = covar_temp*2;
          counter = counter+1;
      end
      
      lik = obs_diff*(covar\transpose(obs_diff)) + log(det(covar_temp))  - ntest*log(2*counter);
      prob = -0.5*lik - ntest/2*log(2*pi);
    end
    
    %)
    %-------------------------------- get --------------------------------
    %
    %  Get a requested member variable.
    %
    %(
    function mval = get(obj,mfield)  
      
      switch(mfield)
        case {'data'}
          mval = obj.data;
        case {'observations','obs'}
          mval = obj.obs;
        case {'mean_vec','mean'}
          mval = obj.mean_vec;
        case {'K'}
          mval = obj.K;
      end
      
    end  
    %)      
    
  end
  
end

