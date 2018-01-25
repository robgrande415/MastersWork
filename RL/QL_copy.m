function [ params,gpr,theta ] = QL_copy( params_in,gpr_in,theta_in,newMu)
%QL_COPY Summary of this function goes here
%   Detailed explanation goes here
params = params_in;
approximation_on = params.approximation_on;
max_points = params.N_budget;
x_initial = params.s_init;
%u_initial = params.a_init;
n_state_dim = params.N_state_dim;

if approximation_on==3 || approximation_on== 7%GP
        params.N_phi_s = 1;
        params.N_phi_sa = params.N_phi_s*params.N_act; % Number of state-action features
        bandwidth =  params.rbf_mu;
        wn = params.sigma;
        tol = params.tol;
        params.max_points=max_points;
        params.sparsification=1;%1=KL divergence, 2=oldest
        
        params.state_action_slicing_on=1;
        gpr = onlineGP_RL_mod(bandwidth,wn,max_points,tol,params);
        if nargin > 3
            gpr.copy(gpr_in,newMu); %onlineGP_RL(bandwidth,wn,max_points,tol,params);
        else
            gpr.copy(gpr_in); %onlineGP_RL(bandwidth,wn,max_points,tol,params);
        end
        %gpr = onlineGP_RLPE(bandwidth,wn,max_points,tol,params);
        %x_input=[x_initial;u_initial];
        %initialize GP
        %gpr.process(x_input,meas_reward,meas_Qmax,params);
        %[mean_post var_post] = gpr.predict(x_input);
        %                params.N_phi_s=gpr.get('current_size'); 
        theta = theta_in;%since this is not relevant for GPs
    


end

