function [params,gpr,theta] = QL_reset(params)
approximation_on = params.approximation_on;
max_points = params.N_budget;
x_initial = params.s_init;
%u_initial = params.a_init;
n_state_dim = params.N_state_dim;
% Reset The Q function
    if approximation_on==0
        params.N_phi_s = params.N_state; % Number of state-features (= N_state for tabular, equal to RBF s for approx)
        params.N_phi_sa = params.N_phi_s*params.N_act;
        params.state_action_slicing_on=1;
        gpr = onlineGP_RL_mod(0,0,0,0,params);
        theta = zeros(params.N_phi_sa,1);
    elseif approximation_on==1 || approximation_on==6
        %load rbf_c_tabular
        %[X,Y] = meshgrid(params.x_grid,params.y_grid);
        %rbf_c = [X(:)';Y(:)'];
        rbf_c = params.statelist';
        %rbf_c = reshape(params.statelist,params.N_state*n_state_dim,1);
        params.N_phi_s = max(size(rbf_c))+1;% equal to RBF s for approx, 1 for bias
        rbf_mu = ones(params.N_phi_s,1)*params.rbf_mu(1);
        params.rbf_c=rbf_c;
        params.rbf_mu=rbf_mu;
        params.bw=1; %RBF bias
        params.N_phi_sa = params.N_phi_s*params.N_act; % Number of state-action features
        params.state_action_slicing_on=1;
        gpr = onlineGP_RL_mod(0,0,0,0,params);
        if approximation_on==1
            theta = zeros(params.N_phi_sa,1);
        elseif approximation_on==6
            theta = zeros(params.N_phi_sa,1);
            params.w = theta;
        end
    elseif approximation_on==3 || approximation_on== 7%GP
        params.N_phi_s = 1;
        params.N_phi_sa = params.N_phi_s*params.N_act; % Number of state-action features
        bandwidth =  params.rbf_mu;
        wn = params.sigma;
        tol = params.tol;
        params.max_points=max_points;
        params.sparsification=1;%1=KL divergence, 2=oldest
        params.state_action_slicing_on=1;

        
        gpr = onlineGP_RL_mod(bandwidth,wn,max_points,tol,params);
        %x_input = [x_initial;u_initial];
        %gpr.process(x_input,0,0,params);
        
        %gpr = onlineGP_RLPE(bandwidth,wn,max_points,tol,params);
        %x_input=[x_initial;u_initial];
        %initialize GP
        %gpr.process(x_input,meas_reward,meas_Qmax,params);
        %[mean_post var_post] = gpr.predict(x_input);
        %                params.N_phi_s=gpr.get('current_size'); 
        theta = 0;%since this is not relevant for GPs
    elseif approximation_on==2||approximation_on==4 %BKR-CL
        params.dictionary_size = max_points;       
        rbf_c = zeros(params.N_state_dim,params.dictionary_size);
        params.o = skernel(rbf_c',params.sigma,params.dictionary_size,params.tol);
        params.rbf_c = params.o.get_dictionary();%dictionary of RBF centers
        params.rbf_c = params.rbf_c';%transpose
        params.N_phi_s = size(params.rbf_c,2)+1;
        params.N_phi_sa=params.N_phi_s*params.N_act;
        params.n2=size(params.rbf_c,params.N_state_dim);%get n2 based on center allocation get n2 based on center allocation
        theta = zeros(params.N_phi_sa,1);
        rbf_mu = ones(params.N_phi_s,1)*params.mu;
        params.rbf_mu = rbf_mu;
        params.bw=1; %RBF bias
        params.state_action_slicing_on=1;
        gpr = onlineGP_RL_mod(0,0,0,0,params);
        if params.cl_on==1
            stored_data.s_stack = zeros(n_state_dim,max_points);
            stored_data.s_n_stack = zeros(n_state_dim,max_points);
            stored_data.u_stack = zeros(1,max_points);
            stored_data.r_stack = zeros(1,max_points);
            stored_data.cl_learning_rate=zeros(1,max_points);
            stored_data.index=0;
            stored_data.points_in_stack=stored_data.index;
            %             [stored_data]=record_data_in_stack(x_initial*0,s_
            %             new,u_init,0,stored_data.index,params,stored_data
            %             );
            params.stored_data = stored_data;
        end
    elseif approximation_on==5 %GP-hyperparameter learning
             x_input=[x_initial;u_initial];
        %s_next = gridworld_trans(x_initial,u_init,params);
        %[rew,breaker] = gridworld_rew(s_next,params);
        %[Q_opt,action] = gridworld_Q_greedy_act(theta,s_next,params,gpr);
        meas_reward=0;%%rew;
        meas_Qmax=0;%Q_opt;
        
        %initialize GP
        params.gpr.process(x_input,meas_reward,meas_Qmax,params);
        [mean_post var_post] = gpr.predict(x_input);
        var_post = 2-var_post; %slight hack
        %                params.N_phi_s=gpr.get('current_size');
    end
    