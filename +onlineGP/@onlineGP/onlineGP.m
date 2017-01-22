%============================== onlineGP ==================================
%
%  This class implements the sparse online GP algorithm presented in the 
%  reference for basic GP regression with a Gaussian kernel. This version
%  of the online GP algorithm learns the parameters as well.
%
%  This code is currently designed strictly for squared exponential 
%  kernels.
%
%  Reference(s): 
%    Sparse Online Gaussian Processes -Csato and Opper, Tech Report
%    Csato's thesis
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
%============================== onlineGP ==================================
%
%  Name:		onlineGP.m
%
%  Author: 		Hassan A. Kingravi
%
%  Created:  	2011/02/27
%  Modified: 	2013/08/15
%
%============================== onlineGP ==================================
classdef onlineGP < handle 
  % class properties   
  properties (Access = public)    
    % these values are asked by the user in the constructor 
    sigma      = 1;   % Bandwidth for SE kernel.
    noise     = 0.1;  % Noise parameter for SE kernel.
    ncent     = 0;    % Budget for basis vector set. 
    tol       = 1e-5; % Tolerance for kernel linear independence test.     
    
    % these values are not asked from the user 
    A            = 1;                % Amplitude for SE kernel. 
    color_val    = [192,255,62]/255; % Default setting of fill for 1D variance.    
    line_width   = 2.0;              % Line width for 1D plot. 
    marker_size  = 7;                % Marker size for basis vectors in 1D plot. 
    sigma_bound  = 2;                % Sigma bounds for variance. 
    label_size   = 14;               % Font size of x and y labels.         
  end
    
  % hidden variables 
  properties (Access = protected)    
    BV           = [];       % Basis vector set.
    K            = [];       % Kernel matrix.
    alpha        = [];       % Mean parameter.
    C            = [];       % Inverted covariance matrix.
    Q            = [];       % Inverted Gram matrix.
    current_size = [];       % Current size of BV set.
    obs          = [];       % Set of observations used for current estimate.
    jitter       = 1e-5;     % Regularization parameter for kernel matrix.            
    reg_type     = 'normal'; % Type of regularization.
    dim          = [];       % Dimensionality of data.         
  end
  
  % class methods 
  methods
    
    function obj = onlineGP(sigma,noise,ncent,tol)
      %  Constructor for onlineGP.
      %
      %  Inputs:
      %    sigma     - 1 x 1 positive scalar for bandwidth of SE kernel
      %    noise     - 1 x 1 positive scalar for noise of SE kernel
      %    ncent     - 1 x 1 positive scalar for number of centers (budget)
      %    tol       - 1 x 1 positive scalar for tolerance for kernel linear
      %                      independence test
      %
      %  Outputs:
      %    -none      
      try 
      % nesting structure to set new values 
      if nargin >=1
        
        if sigma <= 0
          exception = MException('VerifyOutput:OutOfBounds', ...
                       'Bandwidth must be strictly positive');
          throw(exception);
        end
        
        if ~isempty(sigma)
          obj.sigma = sigma;         
        end 
        
        if nargin >=2          
          if noise < 0
            exception = MException('VerifyOutput:OutOfBounds', ...
                         'Noise must be positive');
            throw(exception);
          end
          
          if ~isempty(noise)
            obj.noise = noise;
          end
        
          if nargin >=3
            if ncent < 0
              exception = MException('VerifyOutput:OutOfBounds', ...
                           'Number of centers must be positive');
              throw(exception);
            end            
            
            if ~isempty(ncent)
              obj.ncent = ncent;
            end
            
            if nargin >=4

              if tol < 0
                exception = MException('VerifyOutput:OutOfBounds', ...
                             'Tolerance must be positive');
                throw(exception);
              end                          
              
              if ~isempty(tol)
                obj.tol = tol;
              end
            end  
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
      
    function process(obj,train_data,y)
      %  Takes in a collection of data and generates an initial Gaussian
      %  process model with the associated kernel matrix, its inversion and
      %  the alpha vector. Currently, it's assumed that only a single point is
      %  passed in to initialize.
      %
      %  Inputs:
      %    data  	 - d x 1 data matrix passed in columnwise
      %    y         - 1 x 1 column vector of observations
      %
      %  Outputs:
      %    -none
      
      %create initial GP model
      obj.BV = train_data;
      obj.obs = y';
      obj.current_size = size(train_data,2);
      obj.dim = size(train_data,1);
      
      I = eye(obj.current_size);
      obj.K = onlineGP.kernel(train_data,train_data,obj.sigma,obj.A);
      obj.Q = (obj.K + obj.jitter.*eye(obj.current_size))\I; % inverted gram matrix
      obj.C = (obj.K + obj.noise*eye(obj.current_size))\I; % less numerically stable than cholesky
      obj.alpha = obj.C*y';
      
    end
    
    function [pred_mean,pred_var] = predict(obj,test_data,covar_type)
      %  Given a new datapoint, predict a new value based on current model
      %
      %  Inputs:
      %    data  	    - d x n data matrix passed in columnwise
      %    covar_type - string: 'full' means return full covariance
      %
      %  Outputs:
      %    pred_mean  - 1 x n vector representing predictive mean evaluated
      %                       on testing data
      %    pred_var   - 1 x n vector representing predictive variance
      %                       evaluated on testing data
      %                 n x n matrix representing predictive covariance
      %                       evaluated on testing data if 'full' flag on
      k = onlineGP.kernel(test_data,obj.BV,obj.sigma,obj.A)';
      pred_mean = k'*obj.alpha;
      
      if nargout > 1                                
        if nargin == 2
          pred_var = onlineGP.kernel(test_data,test_data,obj.sigma,obj.A) + k'*obj.C*k;
          pred_var = transpose(diag(pred_var));
        elseif nargin == 3 && strcmp(covar_type,'full')
          pred_var = onlineGP.kernel(test_data,test_data,obj.sigma,obj.A) + k'*obj.C*k;
        end
      end
      
    end
    
    function update(obj,x,y)
      %  Given a new data pair, update the model.
      %
      %  Inputs:
      %    x  	   - d x 1 data matrix passed in columnwise
      %    y         - 1 x 1 column vector of observations
      %
      %  Outputs:
      %    -none
      %
      %  Note:
      %    reg_type  - optional string
      %                - 'regularize': use jitter factor in computations
      %                - 'normal': don't use regularization
      %                Default is no regularization
      
      % first compute simple update quantities
      %disp(x);
      %disp(y);
      k_t1 = onlineGP.kernel(x,obj.BV,obj.sigma,obj.A)';   % pg 9, eqs 30-31
      noise_x = obj.noise^2 + k_t1'*obj.C*k_t1 + obj.A;
      q_t1 = (y - k_t1'*obj.alpha)/(noise_x + obj.noise^2);
      r_t1 = -1/(noise_x + obj.noise^2);
      
      % compute residual projection update quantities
      e_t1 = obj.Q*k_t1; % residual vector pg 6, eq 16
      gamma_t1 = double(obj.A-k_t1'*e_t1); % novelty of new point w.r.t RKHS: pg 7, eq 23
      eta_t1 = 1/(1+gamma_t1*r_t1); % numerical stability parameter
      
      % checks to see if point is very close to last point
      %min_dist = min(sum(sqrt((obj.BV-repmat(x,1,size(obj.BV,2))).^2),1));
      
      if gamma_t1 < obj.tol        
        % in this case, addition of point to basis doesn't help much, so
        % don't add it, and compute update quantities in terms of old vectors
        % note that data, obs and gram matrix inverse not updated
        s_t1 = obj.C*k_t1 + e_t1;                  %pg 5, eqs 9, but modified
        obj.alpha = obj.alpha + q_t1*eta_t1*s_t1;
        obj.C = obj.C + r_t1*eta_t1*(s_t1*s_t1');
        
      else
        % in this case, you need to add the points
        obj.current_size = obj.current_size + 1;
        
        %in this case, you can simply add the points
        s_t1 = [obj.C*k_t1; obj.A];
        obj.alpha = [obj.alpha; 0] + q_t1*s_t1;
        obj.C = [obj.C zeros(obj.current_size-1,1); zeros(1,obj.current_size)] + r_t1*(s_t1*s_t1');
        
        % update basis vectors and observations
        obj.BV = [obj.BV x];
        obj.obs = [obj.obs; y];
        
        % update Gram matrix and inverse
        obj.K = [obj.K k_t1; k_t1' obj.A]; %GIRISHS CODE
        %K = onlineGP.kernel(BV,BV,sigma); %EDIT BY ROB TO MAKE POS DEF
        
        % EDIT: ROB 2-12, when new points are added with new sigma, no
        % longer PD                
        
        if strcmp(obj.reg_type,'normal')
          obj.Q = obj.K\eye(size(obj.K,1));
        else
          obj.Q = (obj.K + obj.jitter.*eye(obj.current_size))\eye(size(obj.K,1));
        end
        
        if obj.current_size <= obj.ncent
          %do nothing
        else
          % now you must delete one of the basis vectors; follow figure 3.3
          % first, compute which vector is least informative (1), pg 8, eq 27
          scores = zeros(1,obj.current_size);
          for i=1:obj.current_size
            scores(i) = abs(obj.alpha(i))/obj.Q(i,i);
          end
          
          %find index of minimum vector
          [~, index] = min(scores);
          
          %now begin update given in (1), pg 8, eq 25
          
          %first compute scalar parameters
          a_s = obj.alpha(index);
          c_s = obj.C(index,index);
          q_s = obj.Q(index,index);
          
          %compute vector parameters
          C_s = obj.C(:,index);
          C_s(index) = [];
          Q_s = obj.Q(:,index);
          Q_s(index) = [];
          
          %shrink matrices
          obj.alpha(index) = [];
          obj.C(:,index)   = [];
          obj.C(index,:)   = [];
          obj.Q(:,index)   = [];
          obj.Q(index,:)   = [];
          obj.K(:,index)   = [];
          obj.K(index,:)   = [];
          
          %finally, compute updates
          obj.alpha = obj.alpha - (a_s/q_s)*(Q_s);
          obj.C = obj.C + (c_s/(q_s^2))*(Q_s*Q_s') - (1/q_s)*(Q_s*C_s' + C_s*Q_s');
          obj.Q = obj.Q - (1/q_s)*(Q_s*Q_s');
          
          obj.current_size = obj.current_size - 1;
          obj.BV(:,index) = [];
          obj.obs(index) = [];
        end
        
      end
    end
    
    function [fig_handle] = visualize(obj,fig_data,fig_struct)
      %  Given a data array, generate either a 1D plot of the predictive
      %  mean with a sigma plot of the predictive variance, or a 2D plot 
      %  with either a contour or surf plot of either the predictive mean, 
      %  variance, or both. 
      %
      %  Inputs:
      %    fig_data   - d x nsamp data matrix passed in columnwise
      %    fig_struct - structure of parameters 
      %                 Common parameters:
      %                   axes        : array of dimensions to slice along 
      %                                 fig_data for computation
      %                   labels      : 1 x 2 string cell array with x and 
      %                                 y labels 
      %                   fig_desired : 'both'     - plot both mean and 
      %                                              variance                   
      %                                 'mean'     - plot only mean (2D)
      %                                 'variance' - plot only variance (2D)
      %                 2D plots: 
      %                   fig_type    : 'contour' (default)
      %                                 'surf'
      %
      %  Outputs:
      %    fig_handle - handle to generated figure
      %    
      try
        % verify number of arguments, and set defaults for figure creation
        if nargin == 1 % only obj passed in 
          exception = MException('VerifyInput:InsufficientArgs', ...
                        'Must pass in data to visualize');
          throw(exception);
          
        elseif nargin >= 2  
          % check to make sure dimensionality consistent 
          fig_dim = size(fig_data,1);
          if obj.dim ~= fig_dim
            exception = MException('VerifyInput:IncorrectArgs', ...
                         'Inconsistent dimensionality');
            throw(exception);
          end
          
          if nargin == 2            
            % in this case, only data is available 
            if fig_dim == 1
              axes = 1;
              labels{1} = 'x';
              labels{2} = 'y';
              fig_desired = 'both';
            else
              % nothing to reshape, so throw exception
              exception = MException('VerifyInput:IncorrectArgs', ...
                'For 2D plot, user must specify nrows and ncols for meshgrid');
              throw(exception);              
            end
          elseif nargin == 3                        
            % in this case, we need to check values in figure_struct
            
            if isfield(fig_struct,'labels')
              if length(fig_struct.labels) == 2 && iscell(fig_struct.labels)
                labels = fig_struct.labels;
              else
                disp('Incorrect length or type for labels array: reverting to defaults.')
                labels{1} = 'x';
                labels{2} = 'y';
              end
            else
              % use default for labels
              labels{1} = 'x';
              labels{2} = 'y';
            end
              
            if fig_dim == 1
              % check if fields exist and have the appropriate length
              if isfield(fig_struct,'axes')
                if length(fig_struct.axes) == 1
                  axes = fig_struct.axes; 
                else
                  disp('Incorrect length for axes array: reverting to defaults.')
                  axes = 1;
                end
              else 
                % use default for axes 
                axes = 1;
              end
              
              if isfield(fig_struct,'labels')
                if length(fig_struct.labels) == 2 && iscell(fig_struct.labels)
                  labels = fig_struct.labels;
                else
                  disp('Incorrect length or type for labels array: reverting to defaults.')                  
                  labels{1} = 'x';
                  labels{2} = 'y';
                end
              else
                % use default for labels
                labels{1} = 'x';
                labels{2} = 'y';
              end  
                
              % always display variance for 1D plot
              fig_desired = 'both';
            else              
              % Must perform particularly strict type checking for nrows
              % and ncols 
              if ~isfield(fig_struct,'nrows') || ~isfield(fig_struct,'ncols')
                exception = MException('VerifyInput:InsufficientArgs', ...
                             'For 2D plot, user must specify nrows and ncols for meshgrid');
                throw(exception);
              elseif ~isnumeric(fig_struct.nrows) || ~isnumeric(fig_struct.ncols)
                exception = MException('VerifyInput:InsufficientArgs', ...
                             'nrows and ncols must be positive integers');
                throw(exception);                
              elseif fig_struct.nrows < 1 || fig_struct.ncols < 1                
                exception = MException('VerifyInput:InsufficientArgs', ...
                             'nrows and ncols must be positive integers');
                throw(exception);                                
              end
              
              % check if fields exist and have the appropriate length
              if isfield(fig_struct,'axes')
                if length(fig_struct.axes) == 2
                  axes = fig_struct.axes; 
                else
                  disp('Incorrect length for axes array: reverting to defaults.')
                  axes = [1,2];
                end
              else 
                % use default for axes 
                axes = [1,2];
              end
                                           
              if isfield(fig_struct,'fig_type')
                fig_type = fig_struct.fig_type; 
                
                if ~ischar(fig_type)
                  disp('Incorrect type for fig_type: reverting to defaults.')
                  fig_type = 'contour';
                elseif ~strcmp(fig_type,'surf') &&~strcmp(fig_type,'contour')
                  disp('Incorrect value for fig_type: reverting to defaults.')
                  fig_type = 'contour';
                end  
                
              else
                % use default for figure type 
                fig_type = 'contour';
              end
              
              % check which plots are desired
              if isfield(fig_struct,'fig_desired')
                fig_desired = fig_struct.fig_desired; 
                
                if ~ischar(fig_desired)
                  disp('Incorrect type for fig_desired: reverting to defaults.')
                  fig_desired = 'both';
                elseif ~strcmp(fig_desired,'mean') && ~strcmp(fig_desired,'variance') ...
                         && ~strcmp(fig_desired,'both')
                  disp('Incorrect value for fig_type: reverting to defaults.')
                  fig_desired = 'both';
                end  
              else
                
                fig_desired = 'both';
              end
                            
            end     
           
          end  
                             
        end
        
        % once arguments are processed, plot figure
        if strcmp(fig_desired,'both')
          [pred_mean,pred_var] = predict(obj,fig_data(axes,:));        
        elseif strcmp(fig_desired,'mean') % more efficient 
          [pred_mean] = predict(obj,fig_data(axes,:));   
        elseif strcmp(fig_desired,'variance') % more efficient 
          [~,pred_var] = predict(obj,fig_data(axes,:));           
        end        
                        
        if fig_dim == 1          
          x_axis = fig_data(axes,:);
          
          figure;
          f = [pred_mean+obj.sigma_bound*sqrt(pred_var');...
                flipdim(pred_mean-obj.sigma_bound*sqrt(pred_var'),1)];
          fill([x_axis'; flipdim(x_axis',1)], f, obj.color_val)
          hold on;          
          fig_handle = plot(x_axis, pred_mean','k','LineWidth',obj.line_width);
          hold on; 
          plot(obj.BV, obj.obs, 'ro','MarkerSize',obj.marker_size);
          legend([num2str(obj.sigma_bound) '\sigma'],'mean','basis vectors')
          xlabel(labels{1},'FontSize',obj.label_size)
          ylabel(labels{2},'FontSize',obj.label_size)
                    
        elseif fig_dim == 2
          X = reshape(fig_data(1,:),fig_struct.nrows,fig_struct.ncols);
          Y = reshape(fig_data(2,:),fig_struct.nrows,fig_struct.ncols);
          
          if strcmp(fig_desired,'both')            
            pred_mean = reshape(pred_mean,fig_struct.nrows,fig_struct.ncols);
            pred_var = reshape(pred_var,fig_struct.nrows,fig_struct.ncols);

            fig_handle = subplot(1,2,1);
            if strcmp(fig_type,'surf')             
              surf(X,Y,pred_mean);
            else
              contour(X,Y,pred_mean);
            end
            title('Predictive Mean')
            xlabel(labels{1},'FontSize',obj.label_size)
            ylabel(labels{2},'FontSize',obj.label_size)
            
            subplot(1,2,2);              
            if strcmp(fig_type,'surf')             
              surf(X,Y,pred_var);
            else
              contour(X,Y,pred_var);
            end            
            title('Predictive Variance')
            xlabel(labels{1},'FontSize',obj.label_size)
            ylabel(labels{2},'FontSize',obj.label_size)        
            
          elseif strcmp(fig_desired,'mean')
            pred_mean = reshape(pred_mean,fig_struct.nrows,fig_struct.ncols);
            if strcmp(fig_type,'surf')             
              surf(X,Y,pred_mean);
            else
              contour(X,Y,pred_mean);
            end
            title('Predictive Mean')
            xlabel(labels{1},'FontSize',obj.label_size)
            ylabel(labels{2},'FontSize',obj.label_size)
            
          elseif strcmp(fig_desired,'variance')
            pred_var = reshape(pred_var,fig_struct.nrows,fig_struct.ncols);
            if strcmp(fig_type,'surf')             
              surf(X,Y,pred_var);
            else
              contour(X,Y,pred_var);
            end
            title('Predictive Variance')
            xlabel(labels{1},'FontSize',obj.label_size)
            ylabel(labels{2},'FontSize',obj.label_size)
            
          end          
          
        else 
          exception = MException('VerifyInput:IncorrectArgs', ...
                         'Unknown figure type');
          throw(exception);
        end
        
      catch ME
        disp([ME.message '!'])  
        err = MException('VerifyInput:IncorrectArgs', ...
                'Exiting due to incorrect inputs.');
        throw(err);        
      end
    end  

    function mval = get(obj,mfield)
      %
      %  Get a requested member variable.
      %
      switch(mfield)
        case {'basis','BV'}
          mval = obj.BV;
        case {'obs'}
          mval = obj.obs;
        case {'K','kernel'}
          mval = obj.K;
        case {'Q'}
          mval = obj.Q;
        case {'C'}
          mval = obj.C;
        case {'current_size','size','current size'}
          mval = obj.current_size;
        case{'full'}
          full = (obj.current_size >= obj.ncent);
          mval = full;
        case{'sigma','bandwidth'}
          mval = obj.sigma;
        case{'noise','snr'}
          mval = obj.noise;
        case{'jitter'}
          mval = obj.jitter;   
        case{'reg_type'}
          mval = obj.reg_type;   
      end
      
    end
    
    function set(obj,mfield,mval)
      %
      %  Set a requested member variable.
      %      
      switch(mfield)
        case {'basis','BV'}
          obj.BV = mval;
        case {'obs'}
          obj.obs = mval;
        case {'K','kernel'}
          obj.K = mval;
        case {'Q'}
          obj.Q = mval;
        case {'C'}
          obj.C = mval;
        case {'current_size','size','current size'}
          obj.current_size = mval;
        case{'sigma','bandwidth'}
          obj.sigma = mval;
        case{'noise','snr'}
          obj.noise = mval;
        case{'alpha'}
          obj.alpha = mval;
        case{'reg_type'}
          obj.reg_type = mval;          
      end
      
    end
    
   
    function copy(obj,new_gp)
      %
      %  Copy old GP into a newly instantiated GP.
      %      
      new_gp.set('BV',obj.BV);
      new_gp.set('obs',obj.obs);
      new_gp.set('alpha',obj.alpha);
      new_gp.set('C',obj.C);
      new_gp.set('K',obj.K);
      new_gp.set('Q',obj.Q);
      new_gp.set('current_size',obj.current_size);
      new_gp.set('sigma',obj.sigma);
      new_gp.set('noise',obj.noise);
    end
    
    function save(obj,model_name)
      %
      %  Save the current model. Takes as input a string for the filename that
      %  the model is saved to.
      %
      
      saved_BV = obj.BV;
      saved_K = obj.K;
      saved_alpha = obj.alpha;
      saved_C = obj.C;
      saved_Q = obj.Q;
      saved_current_size = obj.current_size;
      saved_obs = obj.obs;
      
      save(model_name,'saved_BV','saved_K','saved_alpha','saved_C', ...
        'saved_Q','saved_current_size','saved_obs');
    end
    
    function load(obj,sBV,sK,sA,sC,sQ,sCS,sO)
      %
      %  Load a model from a file. Takes as input a string for the filename
      %  that the model is saved to.
      %
      obj.BV = sBV;
      obj.K = sK;
      obj.alpha = sA;
      obj.C = sC;
      obj.Q = sQ;
      obj.current_size = sCS;
      obj.obs = sO;
    end      
    
  end
    
    
end