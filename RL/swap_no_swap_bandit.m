clear;
%addpath('C:\cygwin\home\Tom\mit\gpq\GPQlearning');
%savefile = ['tmp','m',num2str(nm),'bw',num2str(nbw),'am',num2str(na),'nn',num2str(nn),'bg',num2str(nbg),'to',num2str(nt)];
%addpath('C:\Miao Liu\svn2\code\GPQlearning\gridworld\cluster\GP\Gridsearch');

%want to test: 
%1) mean backup with eps-greedy -TODO
%2) mean backup with mean greedy
%3) mean backup with variance action selection
%4) var backup with variance action seection
%5) var backup with mean greedy selection

%variations for doing the updates:
%updateStyle = 1; %total copy of GP2
updateStyle = -1; %replay the basis vectors of GP2 in GP1 if they bring it down and have high confidence
epsilon1 = 0.05; %threshold basis points needs to come down by before we do a check


%GP params

params.approximation_on = 3;
gpbandwidth = 0.5;
gpamptitude = 1.0;%0.5; %NOTE: This parameter is controlling the initial variance estimate %1
gpnoise = 0.1; 
tol  = 0.01; %10^-2;
params.A = gpamptitude;
params.sigma = gpnoise;
params.rbf_mu= ones(1,1)*gpbandwidth; %bandwidth of GP
params.tol = tol;
params.N_budget=1;
params.epsilon_data_select=0.2;
params.cl_on = 0;

RMAX = 0.0;  %we're going to subtract this from all rewards to make an equivalent mdp with prior = 0;

firstCopy =1;

%domain params
params.gamma = 0.9;

params.N_state = 1;
N_state = params.N_state;
params.statelist = [1];
params.N_state_dim= 1; %2;
params.N_act = 2;
params.s_init = 1; %[1;1]; % Start in the top left corner
params.a_init = 1;



%experiment parameters
N_eps_length =1; % Length of an episode
N_eps = 200; %200 Number of episodes
N_exec = 10; %20;


rand('state',1000*sum(clock));


     
     rews = zeros(N_exec,N_eps);   
        
for ex=1:N_exec

    %create history storage
    %SART = zeros(N_eps * N_eps_length, params.N_state_dim+3);
    %actIndex = params.N_state_dim+1;
    %rewIndex = params.N_state_dim+2;
    %terIndex = params.N_state_dim+3;

    train_Reward = zeros(1,N_eps);
     train_Steps = zeros(1,N_eps);
     
[params,gpr1,theta1] = QL_reset(params); %need to initialize this with all the variances at 0?

[params,gpr2,theta2] = QL_reset(params);  %second GP actually gets data

if(updateStyle == -1)
    gpr1 = gpr2;
end

step_counter = 0;
rew_exec_ind = [];
%eval_counter = 0;

allCount = 0;
for j = 1:N_eps
    % Reset the initial state
    s_old = params.s_init;
    s_indx_old = 1;%DiscretizeState(s_old',statelist);
    disp(['episode:',num2str(j)]);
    for k = 1: N_eps_length
        %fprintf('step %d  %d   %d\n', k, s_old(1), s_old(2));
        % Is it evlauation time ?
        % Increment the step counter
        allCount = allCount + 1;
        %Greed is good.	
        step_counter = step_counter + 1;
        
        %not using the variance here, because things will be set
        %optimistically already
        [Q_opt,action_indx] = Q_greedy_act_mean(theta1,s_old,params,gpr1);
        action = action_indx;
        % Next State
        
        s_new = twoArmedBandit_trans(s_old,action,params);
        %s_indx_new = DiscretizeState(s_new',statelist);
        %Calculate The Reward
        [rew,breaker] = twoArmedBandit_rew(s_old,action,params);
        rew = rew - RMAX;
        
        disp(['action:',num2str(action)]);
        
        %fprintf('reward was %f', rew);
        
        %if breaker
        %    SART(alCount,:) = [s_old,action,rew,1];
        %else
        %    SART(alCount,:) = [s_old,action,rew,0];
            
        train_Reward(1,j) = train_Reward(1,j) + params.gamma^(k-1)*rew;
        train_Steps(1,j) = k;
        
        
        
        [theta2,gpr2,params,y] = QL_update(step_counter,k,params.approximation_on,theta1,s_new,s_old,s_indx_old,rew,action_indx,params,gpr2,breaker,gpr1);
        [act_val2,act_val_var2] =  Q_value(theta2,s_old,action_indx,params,gpr2);
        [act_val1,act_val_var1] =  Q_value(theta1,s_old,action_indx,params,gpr1);
        %fprintf('state %d %d (%d) gp1: %f  %f gp2 %f %f \n',s_old(1), s_old(2), action_indx, act_val1, act_val_var1, act_val2, act_val_var2);
        if(isnan(act_val2))
            2+2;
        end
        %update GP2
        
        
        if(act_val2 + act_val_var2 < act_val1 - act_val_var1)
            %TODO: The check needs to look at the GPs more carefully and
            %make sure the means are not going up.
            firstCopy = 0;
            %fprintf('Doing a non-stat check\n');
            theta1 = theta2;
            if(updateStyle == 1)
                [params,gpr1,theta1] = QL_copy(params,gpr2,theta2);      
                gpr2.reinitCovar(params);
            
            elseif(updateStyle ==3) %KL divergence check on basis vectors      
                SIG = params.rbf_mu;                SNR = params.sigma;
                %collect the basis vectors
                
                cs1 = gpr1.get('current_size');
                cs2 = gpr2.get('current_size');
                
                %get values from the two GPs
                for jj=1:params.N_act
                %use the correct action
                    
                    %TODO: check if not full BV (look at current store)
                    %TODO: Indexing of actions correctly
                    
                    
                    csa1 = cs1(jj);
                    csa2 = cs2(jj);
                    total = csa1 + csa2;
                    
                    %Checked: here is where the actions are added (*jj)
                    B = ones(params.N_state_dim +1,total) * jj; 
                    
                    BV1 = gpr1.getOnlyAct('BV_store',jj);
                    BV2 = gpr2.getOnlyAct('BV_store',jj); 
                    
                    %Checked: took care of non-full BV and correct indexing
                    %of actions
                    B(1:params.N_state_dim,1:csa1) = BV1(:,1:csa1);  %(:,((jj-1)*params.N_budget+1):((jj-1)*params.N_budget+csa1));  
                    %offset = params.N_budget * params.N_act;
                    B(1:params.N_state_dim,(csa1+1):total) =BV2(:,1:csa2);%BV(:,(offset+(jj-1)*params.N_budget+1):(offset+(jj-1)*params.N_budget+csa2));    
                    
                    B = unique(B','rows')'; %Checked: Eliminate duplicates
                    
                    [mu_IS2, S_IS2] =  gpr2.predict(B,params);
                   [mu_IS1, S_IS1] =  gpr1.predict(B,params);
                        
                   S_IS1 = S_IS1 + eye(size(S_IS1))*SNR^2;
                   S_IS2 = S_IS2 + eye(size(S_IS2))*SNR^2;     
                        
                   detect_size = size(B,2);
                   %this should be probability in GP1, and treat the
                   %predictions from GP2 like they are the real data
                   prob_in_set = -1/2* ((mu_IS2'-mu_IS1')*S_IS1^-1*(mu_IS2-mu_IS1) + log( det(S_IS1)))- detect_size/2*log(2*pi) ;
               
                        
                  %this should be probability in GP2, but if the data is
                  %from GP2, this is really a constant
                   prob_not_in_set = -1/2* ((mu_IS2'-mu_IS2')*S_IS2^-1*(mu_IS2-mu_IS2) + log( det(S_IS2)))- detect_size/2*log(2*pi) ;
                        
               
                        KL_div = 1 / detect_size *(prob_not_in_set-prob_in_set);
                        
                        if(KL_div > epsilon1)
                            
                            fprintf('significant change for action %d!  Making the swap!!!!!!!!!!!!!!!!!!!!\n',jj);
                            [params,gpr1,theta1] = QL_copy(params,gpr2,theta2);
                            gpr2.reinitCovar(params);
                            
                            break;
                        else
                          % fprintf('No big deal, %f  no swap.\n', KL_div);
                        end
                end
            
            
        end
       end       
        
	% if reachs the goal breaks the episode
        if breaker
            break;
        end
        
        s_old = s_new;
        
    end
    
    %alpha = gpr.get('alpha_store');
    %if j>1
    %    diff_alpha_norm(j-1) = norm(alpha-alpha_old,'fro');
    %end
    %alpha_old = alpha;
   
end


%filename = ['result_', num2str(ex), 'backup', num2str(backup), '_select', num2str(select)];  %'result_m',num2str(nm),'bw',num2str(nbw),'am',num2str(na),'nn',num2str(nn),'bg',num2str(nbg),'to',num2str(nt),'r',num2str(i)];
%save(filename,'train_Reward','train_Steps','V','diff_alpha_norm');
rews(ex,:) = train_Reward(1,:);%train_Reward(1,:);

end

mean_Reward = mean(rews);
std_Dev_Reward = std(rews);

filename = ['swap'];  %'result_m',num2str(nm),'bw',num2str(nbw),'am',num2str(na),'nn',num2str(nn),'bg',num2str(nbg),'to',num2str(nt),'r',num2str(i)];
if(updateStyle == -1)
    filename = ['no_swap'];
end
if(updateStyle == 1)
    filename = ['aggressive_swap'];
end
if(updateStyle == 3 && epsilon1 >=0.1 )
    filename = ['cautious_swap'];
end



save(filename,'mean_Reward', 'std_Dev_Reward', 'rews');

