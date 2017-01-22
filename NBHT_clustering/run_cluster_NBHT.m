function out = run_cluster_NBHT(x,y,params,predict_on)
%======================= gp_clustering_simple_script ======================
%  
%  This code tests clustering with Gaussian processes, based on a
%  formulation by Robert Grande. The main class used is GPCluster, which
%  builds GP clusters on the fly based on the following structure: 
%  
%  The basic structure of the code is as follows:
%   a) An online GP object is initialized, and trained on data 
%      from (possibly) different models. 
%   b) At each time step, the online GP's confidence in its estimate of the
%      data is checked. A running list of least probable points (LP) is 
%      kept. 
%   c) Once LP reaches a predefined size, a Gaussian process model is
%      learned from it. The likelihood of the least probable data is
%      computed for both existing GP models. 
%   d) The empirical KL divergence between the models is computed: if this
%      divergence is large enough, the new model needs to be initialized.
%
%  See the reference for more information.
%
%  In this current script, data is fed in from two models. 
%
%  References(s):
%    Nonbayesian Hypothesis Testing with Bayesian Nonparametric Models
%        - Robert Grande
%
%======================= gp_clustering_simple_script ======================
%
%  Name:	 gp_clustering_simple_script_KL.m
%
%  Author: 	 Robert Grande
%  Modifier: Hassan A. Kingravi
%
%  Created:  (?)
%  Modified: 2013/07/10
%
%======================= gp_clustering_simple_script ======================

% add path to onlineGP folder and data
addpath('../')
addpath('../data')
MAX_ITER = size(x,2);
% store current model

actual_model = zeros(1,MAX_ITER);

if nargin < 4
    predict_on = 0;
end
est_model    = zeros(1,MAX_ITER);
roll_backed  = zeros(1,MAX_ITER);
kl_vals      = zeros(1,MAX_ITER);
prob_vals    = zeros(1,MAX_ITER);
ind_vals     = zeros(1,MAX_ITER);

% online GP stuff
bandwidth = params.bandwidth; 
tol = params.tol; 
noise = params.noise;
max_points = params.max_points;
detect_size = params.detect_size;
kl_tol = params.kl_tol;
bin_tol = params.bin_tol;
%gp_Detect_Fault = 10;
gpc = onlineGP.GPClusterKL(bandwidth,noise,max_points,tol,detect_size,kl_tol,bin_tol);
gpc.set('reg_type','regularize');

test_x = zeros(size(x));
test_y = zeros(size(y));
mu = zeros(1,length(y));
sigma = mu;
i=1;
act_model = 1;
tic 
while 1
    
    % get point
    ind = i;
    x0 = x(:,ind);
    f0 = y(:,ind);
    
    if mod(ind,10) == 0
        ind;
    end
    
    test_x(:,ind) = x0;
    test_y(:,ind) = f0;
    
    % learn model
    if i == 1
      % if at first time step, initialize GPCluster
      gpc.process(x0,f0);
    else
      [~, ~, kl_val,rb] = gpc.update(x0,f0);     
      
      % get current model
      est_model(i) = gpc.get('current_model'); 
      roll_backed(i) = rb;       
      
      % store KL and probability values
      kl_vals(i) = kl_val;
    end
    
    %predict
    if predict_on
        [m s] = gpc.predict(x0);
        mu(i) = m;
        sigma(i) = s;
    end
    %EDIT: BY ROB, DOWNSAMPLE
    i = i + 1;
    if i > MAX_ITER
      break;
    end
    
    
end

%output stuff
out.gpc = gpc;
out.est_model = est_model;
out.actual_model = actual_model;
out.roll_backed = roll_backed;
out.kl_vals = kl_vals;
out.params = params;
out.mean = mu;
out.stdev = sigma;
toc



return;
model_change_counter = gpc.get('model_size') - 1;
disp(['Model changed ' num2str(model_change_counter) ' times.'])

% plot to where changes in model occurred
roll_back_inds = find(roll_backed == 1);

figure;
plot(actual_model,'g','LineWidth',2.0);
hold on; 
plot(est_model,'b-.','LineWidth',2.0);
legend('actual','estimated')

figure;
plot(kl_vals,'LineWidth',2.0);
title('KL divergence')

% need to plot data; shows how all modes of the plot work 
fig_struct.nrows       = row_vals;
fig_struct.ncols       = col_vals;
fig_struct.fig_type    = 'surf';
fig_struct.fig_desired = 'mean';
fig_struct.labels{1}   = 'WOB';
fig_struct.labels{2}   = 'RPM';

gpc.visualize(x,fig_struct);  
return;
fig_struct.fig_desired = 'variance';
fig_struct.fig_type    = 'contour';
gpc.visualize(x,fig_struct);  

fig_struct.fig_desired = 'both';
gpc.visualize(x,fig_struct);  


end
