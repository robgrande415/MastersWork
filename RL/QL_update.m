function [theta,feature,params,y] = QL_update(step_counter,k,approximation_on,theta,s_new,s_old,s_indx_old,rew,action_indx,params,feature,isgoal,gprstatic)
% list of methods
% 0- QL-tab
% 1- QL-FB
% 2
% 3- GPQ-epsilon
% 4
% 5-GP-hyper not working well
% 6 - GQ
% 7 - GPQ-2var

if approximation_on==4 
        o = feature;
else
        gpr = feature;
end
if approximation_on==3||approximation_on==5||approximation_on==7%GP approx
    if approximation_on==3||approximation_on==5
        [Q_opt,action_max] = Q_greedy_act_mean(theta,s_new,params,gprstatic);
    else
        [Q_opt,action_max] = Q_greedy_act_var(theta,s_new,params,gprstatic);
    end
    meas_reward=rew;
    meas_Qmax=Q_opt;
    if(isgoal)
        meas_Qmax =0.0;
    end
   % if Q_opt >0
   %     meas_Qmax = 0;
   % else
   %     meas_Qmax=Q_opt;%+randn*0.01;%wn
   % end
    xx=[s_old;action_indx];
    y = meas_reward+params.gamma*meas_Qmax;
    gpr.update(xx,meas_reward,meas_Qmax,params);
    %{
    if isgoal
        meas_reward = params.rew_goal;
        meas_Qmax = 0;
        xx=[s_new;action_max];
        gpr.update(xx,meas_reward,meas_Qmax,params);
    end
%}
elseif approximation_on==0%normal no GP
     alpha =  alpha_init/...
        (step_counter)^alpha_dec;
    % Get the feature vector
    %alpha = 0.1;
    phi_s_old_act_old...
        = Q_calculate_feature(s_indx_old,action_indx,params);
    % Calculate The Values
    val_old = Q_value(theta,s_indx_old,action_indx,params);
    [Q_opt,action_max] = Q_greedy_act(theta,s_new,params,gpr);
    val_new = Q_value(theta,s_new,action_max,params);

    TD = (rew + params.gamma*val_new - val_old);
    theta = theta + alpha*(TD.*phi_s_old_act_old);
    y = 0;
elseif approximation_on==1
         alpha =  alpha_init/...
        (step_counter)^alpha_dec;
    % Get the feature vector
    %alpha = 0.1;
    phi_s_old_act_old...
        = Q_calculate_feature(s_old,action_indx,params);
    % Calculate The Values
    val_old = Q_value(theta,s_old,action_indx,params);
    [Q_opt,action_max] = Q_greedy_act(theta,s_new,params,gpr);
    val_new = Q_value(theta,s_new,action_max,params);

    TD = (rew + params.gamma*val_new - val_old);
    theta = theta + alpha*(TD.*phi_s_old_act_old);
y = 0;
elseif approximation_on==4 %BKR
    alpha =  alpha_init/...
        (step_counter)^alpha_dec;
    stored_data = params.stored_data;
    if params.cl_on==1
        if stored_data.index==0
            stored_data.index=1;
            [stored_data]=record_data_in_stack(s_old,s_new,a_init,rew,stored_data.index,params,stored_data);
        end
    end
    
    if mod(k*10,5) == 0
        %these two elements are shifted downward
        %send current state to add_point, decided whether or not to add the
        %current state into the dictionary as rbf center. If flag then added
        %and kicked out ptNr
        mu = params.mu;
        [flag, ptNr, old_x] = o.add_point(s_old');
        
        if(flag > 0)
            %create the new dictionary
            old_x = old_x';
            params.rbf_c = o.get_dictionary();
            params.rbf_c = params.rbf_c';
            params.N_phi_s = size(params.rbf_c,params.N_state_dim)+1;
            params.N_phi_sa=params.N_phi_s*params.N_act;
            params.rbf_mu = mu*ones(params.N_phi_s+1,1);
            new_center=1;
            
            
            if ptNr > 0 && new_center==1
                %add a state to W, or replace the old one
                if params.cl_on==1
                    [stored_data]=record_data_in_stack(s_old,s_new,action_indx,rew,ptNr,params,stored_data);
                end
            elseif ptNr==0 %or add a weight for the point that has been added
                theta = [theta; zeros(params.N_act,1)];
                if params.cl_on==1
                    stored_data.index=stored_data.index+1;
                    stored_data.points_in_stack=min(stored_data.index,max_points);
                    [stored_data]=record_data_in_stack(s_old,s_new,action_indx,rew,stored_data.index,params,stored_data);
                end
            end
        end
    end %end mod if
    % number_kernels(k)=size(params.rbf_c,2);
    % Get the feature vector
    %                 phi_s_old_act_old...
    %                     = DCmotor_Q_calculate_feature(s_old,action_indx...
    %                     ,params);
    %                 % Calculate The Values
    %                 val_old = DCmotor_Q_value(theta,s_old,action_indx,params);
    %                 [Q_opt,action_max] = DCmotor_Q_greedy_act(theta,s_new,params,gpr);
    %                 val_new = DCmotor_Q_value(theta,s_new,action_max,params);
    %                 phi_max=DCmotor_Q_RBF(s_new,action_max,params);%% for diagnostic
    %                 TD = (rew + gamma*val_new - val_old);
    %                 theta = theta + alpha*(TD.*phi_s_old_act_old);
    
    if params.cl_on==1
        cc=calculate_CLQ_concurrent_gradient(theta,params,stored_data);
        cc = params.cl_rate.*cc;
        theta=theta+alpha*cc;
        stored_data.cl_learning_rate=max(stored_data.cl_learning_rate/cl_rate_decrease,0.000001);
    end
    %                 iter = iter + 1;
    %                 if ~mod(iter, 5)
    %                     for iii = 1 : min(length(theta),15)
    %                     subplot(3,5,iii), plot(iter,theta(iii),'o-'),  hold on
    %                     title(['BKCL Q(s=', num2str( iii - (ceil(iii/5)-1)*5), ',a=', num2str(ceil(iii/5)),')']);
    %                     xlabel('steps');
    %                     end
    %                 end
    %                 if convergence_diagnostic
    %                     E_pi_phi=phi_s_old_act_old*phi_s_old_act_old'+E_pi_phi;
    %       %             E_pi_phi_m=phi_max*phi_max'+E_pi_phi_m;
    %                 end
elseif approximation_on==6
    % alpha =  alpha_init/...
    %    (step_counter)^alpha_dec;
    % Get the feature vector
   alpha =  alpha_init/...
                    (step_counter)^alpha_dec;
    beta =  alpha_init/...
                    (step_counter)^(alpha_dec+0.1);
    phi_s_old_act_old...
        = Q_calculate_feature(s_old,action_indx...
        ,params);
    % Calculate The Values
    val_old = Q_value(theta,s_old,action_indx,params);
    [Q_opt,action_max] = Q_greedy_act(theta,s_new,params,gpr);
    val_new = Q_value(theta,s_new,action_max,params);

    TD = (rew + params.gamma*val_new - val_old);
    phi_s_new_act_new...
        = Q_calculate_feature(s_new,action_max...
        ,params);
    wphi = params.w'*phi_s_old_act_old;

    theta = theta + alpha*(TD.*phi_s_old_act_old - params.gamma*wphi*phi_s_new_act_new);

    params.w = params.w + beta*(TD - wphi).*phi_s_old_act_old;
    y = 0;
    %                 iter = iter + 1;
    %                if ~mod(iter, 100)
    %                    J(iter/100) =  sqrt((TD.*phi_s_old_act_old)'/(phi_s_old_act_old*phi_s_old_act_old'+10^-10*eye(16))*(TD.*phi_s_old_act_old));
    %                     for iii = 1 : 14
    %                     subplot(3,5,iii), plot(iter,theta(iii),'-O'),  hold on
    %                     title(['TDC Q(s=', num2str( iii - (ceil(iii/5)-1)*5), ',a=', num2str(ceil(iii/5)),')']);
    %                     xlabel('steps');
    %                     end
    %                     subplot(3,5,15), plot(iter, J(iter/100),'-O'),  hold on
    %                     title('\sqrt{J}')
    %
    %                 end
%     if convergence_diagnostic_on==1 && approximation_on==1
%         phi_max=DCmotor_Q_RBF(s_new,action_max,params,domain_id);%% for diagnostic
%         E_pi_phi=phi_s_old_act_old*phi_s_old_act_old'+E_pi_phi;
%         E_pi_phi_m=phi_s_old_act_old*phi_max'+E_pi_phi_m;
%     end
end

if approximation_on==4
   feature = o;
else
   feature = gpr;
end