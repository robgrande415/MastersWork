classdef onlineGP_CopyAndReinit < handle & onlineGP.onlineGP
    %ONLINEGP_RECOVAR Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
    end
    
    methods
        function obj = onlineGP_CopyAndReinit(sigma,noise,ncent,tol,amp)
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
        if(nargin >= 5)
            obj.A = amp;
        end
        obj.current_size = 0;
        
        end
        
        
        function [pred_mean,pred_var] = predict(obj,test_data,covar_type)
            if (isempty(obj.current_size) || obj.current_size == 0)
                 pred_mean = zeros(size(test_data,2),1); 
                 pred_var = obj.noise^2 + obj.A;  
            else
                if nargin < 3
                    [pred_mean,pred_var] = predict@onlineGP.onlineGP(obj,test_data);
                else
                    [pred_mean,pred_var] = predict@onlineGP.onlineGP(obj,test_data,covar_type);
                end
            end
        end
        
        function reinitCovar(obj)
            obj.K(1:obj.current_size,1:obj.current_size) = onlineGP.kernel(obj.BV,obj.BV,obj.sigma,obj.A);  %kernel(obj.BV,obj.BV,obj.sigma);
            obj.Q(1:obj.current_size,1:obj.current_size) = obj.K\eye(size(obj.K,1)); 
        end
        
        function replace_BV(obj,x,y)
            %project RBF onto existing BVs
            k = onlineGP.kernel(obj.BV,x,obj.sigma,obj.A);
            K(1:obj.current_size,1:obj.current_size) = onlineGP.kernel(obj.BV,obj.BV,obj.sigma,obj.A);  %kernel(obj.BV,obj.BV,obj.sigma);

        
            delta_alpha = K\k;
       
            %get amplitude of RBF
            mu_k = predict(obj,x);
            mu_k1 = y;
            innovation = mu_k1-mu_k;
            
            %add RBF to function approximation using BVs
            obj.alpha = obj.alpha + delta_alpha*innovation;
            delta_alpha';
            obj.BV;
            obj.alpha';
            %obj.K(1:obj.current_size,1:obj.current_size) = onlineGP.kernel(obj.BV,obj.BV,obj.sigma,obj.A);  %kernel(obj.BV,obj.BV,obj.sigma);
            %obj.Q(1:obj.current_size,1:obj.current_size) = obj.K\eye(size(obj.K,1)); 
        end
        
        function copy(obj,oldGP)
            obj.sigma        = oldGP.sigma;    % Bandwidth for SE kernel.
            obj.noise        = oldGP.noise;  % Noise parameter for SE kernel.
            obj.ncent        = oldGP.ncent;    % Budget for basis vector set. 
            obj.tol = oldGP.tol;   
            obj.A = oldGP.A;
            
            obj.BV           = oldGP.BV;       % Basis vector set.
            obj.K            = oldGP.K;       % Kernel matrix.
            obj.alpha        = oldGP.alpha;       % Mean parameter.
            obj.C            = oldGP.C;       % Inverted covariance matrix.
            obj.Q            = oldGP.Q;       % Inverted Gram matrix.
            obj.current_size = oldGP.current_size;       % Current size of BV set.
            obj.obs          = oldGP.obs;       % Set of observations used for current estimate.
            obj.jitter       = oldGP.jitter;     % Regularization parameter for kernel matrix.            
            obj.reg_type     = oldGP.reg_type; % Type of regularization.
            
        end
        
   end
    
end

