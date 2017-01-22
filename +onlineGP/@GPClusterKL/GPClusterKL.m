%============================= GPClusterKL ================================
%  
%  This class implements Gaussian process regression clustering, based on
%  ideas presented in the reference. The class learns GP models on the fly,
%  based on the samples it receives from the data. The only parameters
%  needed for GPCluster are those to initialize the onlineGP class, 
%  which is just the old onlineGP class without hyperparameter updates.
% 
%  The detection test uses the KL divergence.
%
%  Currently, only the Gaussian kernel is supported. 
% 
%  Dependencies: 
%    -- kernel.m
%    -- gp_regression.m
%    -- onlineGP.m
%    -- least_prob_set.m
%
%  Inputs:
%    bandwidth   - 1 x 1 bandwidth for Gaussian kernel
%    noise       - 1 x 1 noise associated to observations
%    max_points  - 1 x 1 budget for onlineGP algorithm
%    tol         - 1 x 1 tolerance for onlineGP algorithm
%    detect_size - 1 x 1 budget for least probable sets 
%
%  Outputs:
%    gpc        - instantiation of the GPClusterKL class. 
%
%============================= GPClusterKL ================================
%
%  Name:		GPClusterKL.m
%
%  Author: 	 	Hassan A. Kingravi
%  Modifier: 	None
%
%  Created:		2013/07/02
%  Modified:	2013/08/15
%
%
%  Reference(s): 
%    Nonbayesian Hypothesis Testing with Bayesian Nonparametric Models
%        - Robert Grande
%
%============================= GPClusterKL ================================
classdef GPClusterKL < handle
    % class properties   
  properties (Access = public)
    bandwidth        = 1;        % Bandwidth for SE kernel.
    noise            = 0.1;      % Noise parameter for SE kernel.
    max_points       = 0;        % Budget for basis vector set. 
    tol              = 1e-5;     % Tolerance for kernel linear independence test. 
    detect_size      = 0;        % Budget for least probable sets.     
    kl_tol           =  2;       % Tolerance for KL divergence.
    bin_tol          = -0.5;     % Tolerance for inclusion into lps set.  
    reg_type         = 'normal'; % Type of regularization. 
    kl_stack         = zeros(1,2000);
  end
  
  % hidden variables 
  properties (Access = protected)    
    onlineModels     = cell(1);  % List of onlineGP models.
    onlineModelsOld  = cell(1);  % List of old onlineGP models at rollback times.
    onlineModelsOld_Switch = cell(1); %List of old onlineGP models at switching times.
    roll_back_time   = cell(1);  % Roll back times.
    switch_time      = 0;
    onlineLeastProbs = [];       % Least probable set.
    
    internal_time    = 0;        % How many data points have been seen.
    lps_counter      = 0;        % How many data points have been put in the bin.
    rollback_counter = 10;       % Old models lag this many steps behind.
    verbose          = 1;        % Turn on internal messaging. 
    dim              = [];       % Dimensionality of data.
    current_model    = 1;        % Variable indicating current model.
    last_change      = 0;        % Time when model changed last. 
    last_model       = 1;        % Identity of last model. 
  end
  
  methods
    function obj = GPClusterKL(bandwidth,noise,max_points,tol,detect_size,varargin)
      % Constructor for GPClusterKL.
      %
      % Inputs:
      %   bandwidth   - 1 x 1 positive scalar for bandwidth of SE kernel
      %   noise       - 1 x 1 positive scalar for noise for SE kernel
      %   max_points  - 1 x 1 positive integer for number of centers (budget)
      %   tol         - 1 x 1 positive scalar for tolerance for kernel linear
      %                        independence test
      %   detect_size - 1 x 1 positive integer for number of points for
      %                        probability test
      %   kl_tol      - 1 x 1 scalar for KL divergence tolerance
      %   bin_tol     - 1 x 1 scalar for probability bin tolerance
      %
      % Outputs:
      %   -none
      try 
      % nesting structure to set new values 
      if nargin >=1
        
        if bandwidth <= 0
          exception = MException('VerifyOutput:OutOfBounds', ...
                       'Bandwidth must be strictly positive');
          throw(exception);
        end
        
        if ~isempty(bandwidth)
          obj.bandwidth = bandwidth;         
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
            if max_points < 0
              exception = MException('VerifyOutput:OutOfBounds', ...
                           'Number of centers must be positive');
              throw(exception);
            end            
            
            if ~isempty(max_points)
              obj.max_points = max_points;
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
              
              if nargin >= 5                
                
                if detect_size < 0
                  exception = MException('VerifyOutput:OutOfBounds', ...
                    'Detect size must be positive');
                  throw(exception);
                end
                
                if ~isempty(detect_size)
                  obj.detect_size = detect_size;
                  obj.rollback_counter = detect_size;
                end
                
                % finally, set KL and probability tolerances 
                if nargin >= 6                                    
                  obj.kl_tol = varargin{1};                
                  if nargin >= 7
                    obj.bin_tol = varargin{2};
                  end
                end                
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
    
    function process(obj,data,obs)
      %  Initialize first GP model and least probable set.
      %
      %  Inputs:
      %    data       - d x 1 data point passed in columnwise
      %    obs        - 1 x 1 column vector of observation
      %
      % Outputs:
      %   -none      
      gpr = onlineGP.onlineGP_PE(obj.bandwidth,obj.noise,obj.max_points,obj.tol); % if at first time step, initialize GP
      gpr.process(data,obs);
      gpr.set('reg_type',obj.reg_type);
      
      % initialize least probable sets
      obj.dim = size(data,1);
      lps = onlineGP.least_prob_set(obj.detect_size,obj.dim,'Threshold',obj.bin_tol);
      
      % add to models
      obj.onlineModels{1} = gpr;
      obj.onlineModelsOld{1} = gpr;
      obj.onlineModelsOld_Switch{1} = gpr;
      obj.roll_back_time{1}  = 1;
      obj.onlineLeastProbs = lps;
      
    end
    
    function [mean_post,var_post,currentKL, var_model, curr_size] = update(obj,data,obs)      
      % Update models using data.
      %
      % Inputs:
      %   data       - d x 1 data point passed in columnwise
      %   obs        - 1 x 1 column vector of observation
      %
      % Outputs:
      %   mean_post  - 1 x 1 mean prediction of model with lowest KL val
      %   var_post   - 1 x 1 variance of model with lowest KL val
      %   kl_val     - 1 x 1 lowest KL val
      
      obj.internal_time = obj.internal_time + 1; % update internal time
      
      %EDIT: ROB, check models against eachother to see if two models are
      %significantly close together.
      if (obj.internal_time - obj.switch_time) > obj.rollback_counter
          obj.gpc_compare_models();
      end
        
      % store old models
      roll_back_done = 0;
      %obj.internal_time is clock, lps_counter
      
      rollback_ind = mod(obj.internal_time-1,obj.rollback_counter)+1;
      for ii=1:length(obj.onlineModels)
          gp_new = onlineGP.onlineGP_PE(obj.bandwidth,obj.noise,obj.max_points,obj.tol);  % new GP
          gp_new.set('reg_type',obj.reg_type);
          gp_temp = obj.onlineModels{ii};
          gp_temp.copy(gp_new); % copy temp into new
          obj.onlineModelsOld{ii,rollback_ind} = gp_new;
          obj.roll_back_time{ii,rollback_ind} = obj.internal_time;
      end
        
      %old code
      %{
      if mod(obj.internal_time,obj.rollback_counter) == 0
        roll_back_done = 1; 
        for ii=1:length(obj.onlineModels)
          gp_new = onlineGP.onlineGP_PE(obj.bandwidth,obj.noise,obj.max_points,obj.tol);  % new GP
          gp_new.set('reg_type',obj.reg_type);
          gp_temp = obj.onlineModels{ii};
          gp_temp.copy(gp_new); % copy temp into new
          obj.onlineModelsOld{ii} = gp_new;
          obj.roll_back_time{ii} = obj.internal_time;
        end
        
      end
      %}
      
      lps = obj.onlineLeastProbs; % get set of least probable points
      
      % update current model with data and compute likelihood of data arising
      % from this model
      gpr_current = obj.onlineModels{obj.current_model};
      gpr_current.update(data,obs);
      
      %optimize hyperparameters
      if 0
          if length(gpr_current.get('obs')) > 50
            gpr_current.update_param(data,obs);
          end
      end
      [mean_post, var_post] = gpr_current.predict(data);
      var_model = var_post;
      %    is_full = gpr_current.get('max_eig');
      curr_size = gpr_current.get('current_size');
      
      prob_pt = inf;
      if length(gpr_current.get('obs')) >= min([20,0.5*obj.max_points])
        % check likelihood of current point
        prob_pt = log(normpdf(obs,mean_post,sqrt(var_post+gpr_current.get('snr')^2)));
      end
      
      
      %if point is unlikely, at it to the set
      if prob_pt < obj.bin_tol
          obj.lps_counter = obj.lps_counter + 1;
        lps.update(prob_pt,data,obs);
      end
      
      full_set_acquired = lps.get('full');
      
      % compute KL divergence between lps and current model
      currentKL = 0;
      kl_ss = inf; %steady state value of lrt vals
      
      
      if full_set_acquired
        % calculate probability of new set
        SIGMA = gpr_current.get('bandwidth');
        SNR = gpr_current.get('snr');
        
        % to compute the probability that the data is not generated
        % by the existing model, first train a GP model built from
        % the least probable points
        least_prob_pt = lps.get('least_prob_pt');
        least_prob_obs = lps.get('least_prob_obs');
        
        gpr_rbf = onlineGP.gp_regression(SIGMA,SNR);
        gpr_rbf.train(least_prob_pt', least_prob_obs');
        [mu_NIS] = gpr_rbf.predict(least_prob_pt');
        
        % compute likelihood of subset of data arising from new model
        obs_diff_NIS = least_prob_obs'-mu_NIS;
        prob_not_in_set = gpr_rbf.model_prob(least_prob_pt',obs_diff_NIS);
        
        % EDIT: HASSAN: made small change to predict function; the full flag
        % means that the full covariance matrix is returned, whereas no
        % argument means that the diagonal of the covariance is returned
        [mu_IS, S_IS] = gpr_current.predict(least_prob_pt','full');
        
        % compute likelihood of subset of data arising from new model
        obs_diff_IS = least_prob_obs'-mu_IS';
        prob_in_set = gpr_rbf.model_prob_covar(S_IS,obs_diff_IS);
        
        
        % compute empirical KL divergence between two models
        currentKL = 1/obj.detect_size*(prob_not_in_set-prob_in_set);
        %save KL value. if stack is too small, pad with zeros
        obj.kl_stack(obj.internal_time) = currentKL;
        
        %this block finds the average of the lrt values
        %average value since last swap?
        last_switch_ind = find(obj.kl_stack(1:obj.internal_time) == 0,1,'last');
        
        %have we seen enough samples since the switch
        temp = obj.internal_time - obj.detect_size; %lrt window beginning
        temp2 = obj.internal_time - 2*obj.detect_size;%ensures n >=m
        %steady lrt values since beginning of window
        if last_switch_ind < temp2
            kl_ss = mean(obj.kl_stack(last_switch_ind:temp));
        end
        
      end
      
      %save KL value. if stack is too small, pad with zeros
      obj.kl_stack(obj.internal_time) = currentKL;
      if obj.internal_time >= length(obj.kl_stack)
          obj.kl_stack = [obj.kl_stack, zeros(1,2000)];
      end
      
      % time to relearn if KL divergence is too big
      if currentKL > obj.kl_tol+kl_ss
        % search for best possible model to compare to
        best_prob  = -inf;
        best_model = 1;
        
        for ii = 1:length(obj.onlineModels)
          [mu_IS, S_IS] = obj.onlineModels{ii}.predict(least_prob_pt','full');
          S_IS = S_IS + eye(size(S_IS))*SNR^2;
          temp_prob = -1/2* ((least_prob_obs'-mu_IS')*S_IS^-1*(least_prob_obs-mu_IS)...
            + log( det(S_IS)))- obj.detect_size/2*log(2*pi);
          
          if temp_prob > best_prob
            best_model = ii;
            best_prob  = temp_prob;
          end
        end
        if 1
            2
        end
        prob_in_set = best_prob;
        KL_temp = 1/obj.detect_size*(prob_not_in_set-prob_in_set);
        
        
        % restore old model to avoid corruption
        gp_old = onlineGP.onlineGP_PE(obj.bandwidth,obj.noise,obj.max_points,obj.tol);  % new object
        gp_old.set('reg_type',obj.reg_type);
        rollback_ind = rollback_ind +1;
        if rollback_ind > obj.rollback_counter
            rollback_ind = 1;
        end
        gp_temp = obj.onlineModelsOld{obj.current_model,rollback_ind};
        gp_temp.copy(gp_old);
        obj.onlineModels{obj.current_model} = gp_old;
        
        %save models at switch
        for ii=1:length(obj.onlineModels)
            gp_new = onlineGP.onlineGP_PE(obj.bandwidth,obj.noise,obj.max_points,obj.tol);  % new GP
            gp_new.set('reg_type',obj.reg_type);
            gp_temp = obj.onlineModels{ii};    
            gp_temp.copy(gp_new); % copy temp into new
            obj.onlineModelsOld_Switch{ii} = gp_new;        
        end  
      
        if obj.verbose == 1
          r_str = ['Rolling back Model ' num2str(obj.current_model)...
                   ' to time ' num2str(obj.roll_back_time{obj.current_model,rollback_ind})];
          disp(r_str)
          obj.internal_time
        end
        
       
        obj.switch_time = obj.internal_time;
        
          cluster_num = length(obj.onlineModels);
          gpr_new = onlineGP.onlineGP_PE(obj.bandwidth,obj.noise,obj.max_points,obj.tol);
          gpr_new.set('reg_type',obj.reg_type);
          
          % train new GP
          %EDIT: ROB, do we want to add the observed points? we tend to have
          %problems with there being mixing in the bins so when we retrain,
          %our models get mixed.
          gpr_new.process(least_prob_pt(1,:)',least_prob_obs(1));
          
          
          obj.onlineModels{cluster_num+1} = gpr_new;
          obj.onlineModelsOld{cluster_num+1} = gpr_new;
          obj.roll_back_time{cluster_num+1} = obj.internal_time;
          
          cluster_num = cluster_num + 1;
          obj.current_model = cluster_num;
        
        
        % reset lps
        obj.onlineLeastProbs = onlineGP.least_prob_set(obj.detect_size,obj.dim,'Threshold',obj.bin_tol);
      end
      
      % predict
      [mean_post, var_post]  = obj.onlineModels{obj.current_model}.predict(data);
      
    end
    
    function [pred_mean,pred_var] = predict(obj,data)
      % Compute predictions of current model on data.
      %
      % Inputs:
      %   data        - d x n data points passed in columnwise
      %
      %  Outputs:
      %    pred_mean  - 1 x n vector representing predictive mean evaluated
      %                       on testing data
      %    pred_var   - 1 x n vector representing predictive variance
      %                       evaluated on testing data
      %                 n x n matrix representing predictive covariance
      %                       evaluated on testing data if 'full' flag on
      gpr = obj.onlineModels{obj.current_model};
      
      if nargout == 1
        pred_mean = gpr.predict(data);
      else
        [pred_mean, pred_var] = gpr.predict(data);
      end
    end
    
    function [fig_handles] = visualize(obj,fig_data,fig_struct)
      %  Given a data array, generate either a 1D plot of the predictive
      %  mean with a sigma plot of the predictive variance, or a 2D plot
      %  with either a contour or surf plot of either the predictive mean,
      %  variance, or both.
      %
      %  Inputs:
      %    fig_data   - d x nsamp data matrix passed in columnwise
      %    fig_struct  - structure of parameters
      %                  Common parameters:
      %                    axes        : array of dimensions to slice along
      %                                  fig_data for computation
      %                    labels      : 1 x 2 string cell array with x and
      %                                  y labels
      %                    fig_desired : 'both'     - plot both mean and
      %                                               variance
      %                                  'mean'     - plot only mean (2D)
      %                                  'variance' - plot only variance (2D)
      %                  2D plots:
      %                    fig_type    : 'contour' (default)
      %                                  'surf'
      %
      %  Outputs:
      %    fig_handles - array of handles to generated figures
      %
      nmodels = length(obj.onlineModels);
      fig_handles = cell(1,nmodels);
      
      for i=1:nmodels
        figure;
        gpr = obj.onlineModels{i};
        gpr.visualize(fig_data,fig_struct);
      end
      
    end
    
    function gpc_compare_models(obj)
      % Compares the current model to all previous models to see if there 
      % are redundant models.
      jj=1;
      
      %NEED TO FIX: deleting models that are previous to others.
      while jj <=length(obj.onlineModels)
        %skip if current model
        if jj == obj.current_model
          jj= jj+1;
          continue;
        end
        
        % compare at BV
        BV1 = obj.onlineModels{obj.current_model}.get('BV');
        BV2 = obj.onlineModels{jj}.get('BV');
        BV = BV1;
        %BV= [BV1, BV2];
        [mu1, S1] = obj.onlineModels{obj.current_model}.predict(BV,'full');
        [mu2, S2] = obj.onlineModels{jj}.predict(BV,'full');
        
        % get KL divergence
        SIGMA = obj.onlineModels{obj.current_model}.get('bandwidth');
        SNR = obj.onlineModels{obj.current_model}.get('snr');
        gpr_rbf = onlineGP.gp_regression(SIGMA,SNR);
        obs_diff_IS = mu1'-mu1';
        prob_in1 = gpr_rbf.model_prob_covar(S1,obs_diff_IS);
        obs_diff_IS = mu1'-mu2';
        prob_in2 = gpr_rbf.model_prob_covar(S2,obs_diff_IS);
        
        %EDIT: ROB, probin2 got mixed with probin1
        KL_temp = 1/size(BV,2)*(prob_in1-prob_in2);
        
        %if the current model is similar enough, return to other model
        if KL_temp < (obj.kl_tol)
            
            
          if obj.verbose == 1
            d_str = ['Model ' num2str(obj.current_model)...
              ' too similar to Model ' num2str(jj) ': deleting it. Iteration: ' num2str(obj.internal_time)];
            disp(d_str)
            curr_prob = prob_in1
            old_prob = prob_in2
            
          end
          
          obj.onlineModels(obj.current_model) = [];
          obj.onlineModelsOld{obj.current_model} = [];
          obj.roll_back_time{obj.current_model} = [];
          
          %ROB: this code is buggy
          if obj.current_model < jj
            obj.current_model = jj-1;
          else
            obj.current_model = jj;
          end
          %obj.current_model = jj;
          %if current model is old model, we need to roll back to last
           %switch
           % restore old model to avoid corruption
           %{
           gp_old = onlineGP.onlineGP_PE(obj.bandwidth,obj.noise,obj.max_points,obj.tol);  % new object
           gp_old.set('reg_type',obj.reg_type);
           gp_temp = obj.onlineModelsOld_Switch{obj.current_model};
           gp_temp.copy(gp_old);
           obj.onlineModels{obj.current_model} = gp_old;
           
            %}
           % reset lps
           obj.onlineLeastProbs = onlineGP.least_prob_set(obj.detect_size,obj.dim,'Threshold',obj.bin_tol);
           obj.switch_time= obj.internal_time;
           break;
          
        end
        
        
        jj= jj+1;
      end
      
    end
    
    function fval = get(obj,fname)
      % Get function for protected properties. 
      switch (fname)
        case {'models','onlineModels'}
          fval = obj.onlineModels;
        case {'onlineModelsOld'}
          fval = obj.onlineModelsOld;
        case 'onlineLeastProbs'
          fval = obj.onlineLeastProbs;
        case 'model_size'
          fval = length(obj.onlineModels);
        case 'current_model'
          fval = obj.current_model;
        case 'last_model'
          fval = obj.last_model;
        case 'last_change'
          fval = obj.last_change;
        case 'llp'
          fval = obj.onlineLeastProbs;
      end
      
    end
    
    function set(obj,mfield,mval)
      % Set function for protected properties. 
      switch(mfield)
        case{'sigma','bandwidth'}
          obj.bandwidth = mval;
        case{'noise','snr'}
          obj.noise = mval;
        case{'reg_type'}
          obj.reg_type = mval;          
      end
      
    end
    
  
  end
  
end

