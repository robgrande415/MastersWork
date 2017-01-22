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
clc; close all; clear classes; clear all;

% add path to onlineGP folder and data
addpath('../')
addpath('../data')

% load reward models 
load GPClustering_reward_model

% only two models
nsamp = row_vals*col_vals;

% iteration stuff
i = 1;
MAX_ITER = 9000;
CHANGE1 = 2400;
CHANGE2 = 5000;
CHANGE3 = 6500;
CHANGE4 = 8000;

% set random seed
s = RandStream('mcg16807','Seed',100);
RandStream.setGlobalStream(s)

% load functions 
f1_clean = reshape(Z1,1,nsamp); 
f2_clean = reshape(Z2,1,nsamp); 
f3_clean = reshape(Z3,1,nsamp); 

% store current model
actual_model = zeros(1,MAX_ITER);
est_model    = zeros(1,MAX_ITER);
roll_backed  = zeros(1,MAX_ITER);
kl_vals      = zeros(1,MAX_ITER);
prob_vals    = zeros(1,MAX_ITER);
ind_vals     = zeros(1,MAX_ITER);

X1 = reshape(X,1,row_vals*col_vals);
Y1 = reshape(Y,1,row_vals*col_vals);
x = [X1; Y1]; 

% add noise to measurements
snr = 0.05^2;
f1 = f1_clean + sqrt(snr)*randn(1,nsamp);
f2 = f2_clean + sqrt(snr)*randn(1,nsamp);
f3 = f3_clean + sqrt(snr)*randn(1,nsamp);

% plot observations from reward models 
F1 = reshape(f1,row_vals,col_vals);
F2 = reshape(f2,row_vals,col_vals);
F3 = reshape(f3,row_vals,col_vals);

figure(1);
surf(X,Y,Z1)
hold on; 
plot3(X,Y,F1,'ro','LineWidth',2.0)
title('GP1 over state space')

figure(2);
surf(X,Y,Z2)
hold on; 
plot3(X,Y,F2,'ro','LineWidth',2.0)
title('GP2 over state space')

figure(3);
surf(X,Y,Z3)
hold on; 
plot3(X,Y,F3,'ro','LineWidth',2.0)
title('GP3 over state space')

% online GP stuff
bandwidth = 0.25; 
tol = 1e-6; 
noise = sqrt(snr);
parameter_est = 0;
max_points = 40;
detect_size = 20;
kl_tol = 2.5;
bin_tol = -0.5;
%gp_Detect_Fault = 10;
gpc = onlineGP.GPClusterKL(bandwidth,noise,max_points,tol,detect_size,kl_tol,bin_tol);
gpc.set('reg_type','regularize');

test_x = zeros(size(x));
test_y = zeros(size(f1));

tic 
while 1
    
    % get point
    ind = randi(length(x));
    
    %ind = i; 
    x0 = x(:,ind);
    if i < CHANGE1
      f0 = f1(ind);
      act_model = 1; % inefficient, but whatever       
    elseif i < CHANGE2
      f0 = f2(ind);
      act_model = 2; 
    elseif i < CHANGE3        
      f0 = f1(ind);
      act_model = 1;   
    elseif i < CHANGE4  
      f0 = f3(ind);
      act_model = 3;         
    else
      f0 = f1(ind);
      act_model = 1; 
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
      actual_model(i) = act_model;       
      roll_backed(i) = rb;       
      
      % store KL and probability values
      kl_vals(i) = kl_val;
    end
    
    if i >= MAX_ITER
      break;
    end
    
    i = i + 1;
end

toc

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



