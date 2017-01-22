%============================ least_prob_set ==============================
%
%  This class stores the least probable elements as evaluated by a GP
%  model. 
%
%  References(s):
%    Nonbayesian Hypothesis Testing with Bayesian Nonparametric Models
%        - Robert Grande
% 
%  Inputs:
%    budget  	 - the size of the set 
%    dim         - the dimensionality of the data. l
%
%  Outputs:
%    see functions
%
%============================ least_prob_set ==============================
%
%  Name:		least_prob_set.m
%
%  Author: 		Hassan A. Kingravi
%
%  Created:  	2013/06/27
%  Modified: 	2013/06/27
%
%============================ least_prob_set ==============================
function lps = least_prob_set(budget,dim,typeS,bin_threshold)
  least_prob     = ones(budget,1)*inf;
  least_prob_pt  = zeros(budget,dim); 
  least_prob_obs = zeros(budget,1);
  current_size   = 0; 
  full           = 0;
  type =2;
  if nargin < 4
      bin_threshold = -0.5;
  end
  %picks bin selection criterion
  if nargin > 2
      switch typeS
          case{'LPS','LLS'}
              type = 1;
          case{'Threshold','Thresh','ThreshBin'}
              type = 2;
              if nargin == 3
                  bin_threshold = -0.5;
              end
          case{'Window'}
              type = 3;
          otherwise
              type =2;
              bin_threshold = -1;
      end
  end
  lps.update = @lps_update; 
  lps.get    = @lps_get; 

  %------------------------------- update -------------------------------
  %
  %  Update the least probable set. 
  %
  %  Inputs:
  %    prob_data  - 1 x 1 probability of data point
  %    data       - d x 1 data point passed in columnwise
  %    obs        - 1 x 1 column vector of observation
  %
  %(
  function full_flag = lps_update(prob_data,data,obs)
      switch type
          case 1
              % if the point is less probable than the last element, add
              if prob_data < least_prob(end)
                least_prob = [least_prob(1:end-1); prob_data];
                least_prob_pt = [least_prob_pt(1:end-1,:); data'];
                least_prob_obs = [least_prob_obs(1:end-1,:); obs];
                [least_prob, ind] = sort(least_prob);
                least_prob_pt = least_prob_pt(ind,:);
                least_prob_obs = least_prob_obs(ind);
              
                 % every time you update, check current size
                 if current_size < budget
                   current_size = current_size + 1;   
                 end
                
              end  
          case 2
              if prob_data < bin_threshold
                least_prob = [least_prob(2:end);  prob_data];
                least_prob_pt = [least_prob_pt(2:end,:); data'];
                least_prob_obs = [least_prob_obs(2:end,:); obs];
                if current_size < budget
                   current_size = current_size + 1;   
                end
              end
          case 3
                least_prob = [least_prob(2:end);  prob_data];
                least_prob_pt = [least_prob_pt(2:end,:); data'];
                least_prob_obs = [least_prob_obs(2:end,:); obs];
                if current_size < budget
                   current_size = current_size + 1;   
                end
      end
    % set full flag 
    if current_size == budget  
      full = 1;   
    end
    
    full_flag = full;
      
  end

  %)
  %-------------------------------- get --------------------------------
  %
  %  Get a requested member variable.
  %
  %(
  function mval = lps_get(mfield)

  switch(mfield)
    case {'least_prob'}
	  mval = least_prob;
    case {'least_prob_pt'}
	  mval = least_prob_pt;      
    case {'least_prob_obs'}
	  mval = least_prob_obs;      
    case {'full'}
	  mval = full;      
  end

  end

end

