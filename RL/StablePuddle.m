
%Things to add:

%1) change the stat test to be on each individual basis point in GP2
%2) change the test to consider GP1 as no variance and GP2 to have its
%current variance
%3) change the test to just look at upper tail versus mean, be simple
%4) change the copy and test to just iterate over each action, but you do
%need to reinit the covariance bounds on each of the actions.
%5) change the copy to instead insert the basis from gp2 into gp1 by either
%knocking out the closest one or all within a given distance



clear;
countSteps = 0;

fh1 = figure;
fh2 = figure;


%params.env = 'bandit';
params.env = 'puddle';
%params.env = 'grid';

%variations for doing the updates:
updateStyle = 3; 
%-1 = none, 1 = aggressive, 2 = cautious, 3 = swap

plotQ = zeros(5,5);
plotPol = zeros(5,5);
plotPolX = zeros(5,5);
plotPolY = zeros(5,5);


%addpath('C:\cygwin\home\Tom\mit\gpq\GPQlearning');
%savefile = ['tmp','m',num2str(nm),'bw',num2str(nbw),'am',num2str(na),'nn',num2str(nn),'bg',num2str(nbg),'to',num2str(nt)];
%addpath('C:\Miao Liu\svn2\code\GPQlearning\gridworld\cluster\GP\Gridsearch');

%want to test: 
%1) mean backup with eps-greedy -TODO
%2) mean backup with mean greedy
%3) mean backup with variance action selection
%4) var backup with variance action seection
%5) var backup with mean greedy selection


%gp with mean backup
params.approximation_on = 3;
params.cl_on = 0;



 if(strcmp(params.env,'bandit') == 1)
    epsilon1 = 0.05;
    if(updateStyle == 2)
        epsilon1 = 0.1; %threshold basis points needs to come down by before we do a check
    end

%GP params

    gpbandwidth = 0.5;
    gpamptitude = 0.5; %NOTE: This parameter is controlling the initial variance estimate %1
    gpnoise = 0.1; 
    gptol  = 0.01; %10^-2;
    params.A = gpamptitude;
    params.sigma = gpnoise;
    params.rbf_mu= ones(1,1)*gpbandwidth; %bandwidth of GP
    params.tol = gptol;
    params.N_budget=1;
    params.epsilon_data_select=0.2;
    RMAX = 0.0;  %we're going to subtract this from all rewards to make an equivalent mdp with prior = 0;

    params.gamma = 0.9;    
    params.N_state = 1;
    N_state = params.N_state;
    params.statelist = [1];
    params.N_state_dim= 1; %2;
    params.N_act = 2;
    params.s_init = 1; %[1;1]; % Start in the top left corner
    params.a_init = 1;

    N_eps_length =1; % Length of an episode

 elseif(strcmp(params.env,'acrobot')==1)

     %addpath('C:\cygwin\home\Tom\mit\gpq\GPQlearning\acrobot');
     epsilon1 = 0.5;
    if(updateStyle == 2)
        epsilon1 = 1; %threshold basis points needs to come down by before we do a check
    end

%GP params'



    gpbandwidth = 0.3;
    gpamptitude = 1.0; %NOTE: This parameter is controlling the initial variance estimate %1
    gpnoise = 0.1; 
    gptol  = 0.1; %10^-2;
    params.A = gpamptitude;
    params.sigma = gpnoise;
    params.rbf_mu= ones(1,1)*gpbandwidth; %bandwidth of GP
    params.tol = gptol;
    params.N_budget=200;
    %params.epsilon_data_select=0.2;
    RMAX = 0.0;  %we're going to subtract this from all rewards to make an equivalent mdp with prior = 0;

    params.gamma = 0.95; %0.9    
    %params.N_state = 1;
    %N_state = params.N_state;
    %params.statelist = [1];
    params.N_state_dim= 4; %2;
    params.N_act = 3;
    params.s_init = [0;0;0;0]; %[1;1]; % Start in the top left corner
    params.a_init = 2;

    N_eps_length =250; % Length of an episode
    countSteps = 1;
 elseif(strcmp(params.env,'grid') == 1)
    epsilon1 = 0.1;
    if(updateStyle == 2) %cautious
        epsilon1 = 0.5; %threshold basis points needs to come down by before we do a check
    end

    
    gpbandwidth = 0.5;
    gpamptitude = 0.5; %NOTE: This parameter is controlling the initial variance estimate %1
    gpnoise = 0.1; %10^-1;
    tol  = 0.01; %10^-2;
    params.A = gpamptitude;
    params.sigma = gpnoise;
    params.rbf_mu= ones(1,1)*gpbandwidth; %bandwidth of GP
    params.tol = tol;
    params.N_budget=25;
    params.epsilon_data_select=0.2;
    params.N_obstacle = 0;
    params.obs_list = [];

    RMAX = 0.0;  %we're going to subtract this from all rewards to make an equivalent mdp with prior = 0;

    %domain params
    params.gamma = 0.9;

    N_grid = 5;
    params.N_state = N_grid*N_grid;
    N_state = params.N_state;
    [X,Y] = meshgrid(1:N_grid,1:N_grid);
    params.statelist = [X(:),Y(:)];
    params.N_grid = 5;
    params.s_goal = [N_grid;N_grid]; 
    params.rew_goal = 0;
    params.rew_obs = -1;
    params.N_state_dim= 2; %2;
    params.N_act = 5;
    params.noise = 0.0;%0.1;
    params.s_init = [1;1]; %[1;1]; % Start in the top left corner



    %experiment parameters
    N_eps_length =200; % Length of an episode
elseif(strcmp(params.env,'puddle') == 1)
    epsilon1 = 0.5;
    params.N_grid = 5;
    if(updateStyle == 2) %cautious
        epsilon1 = 0.5; %threshold basis points needs to come down by before we do a check
    end

    
    gpbandwidth = 0.1;
    gpamptitude = 0.5; %NOTE: This parameter is controlling the initial variance estimate %1
    gpnoise = 0.1; %10^-1;
    tol  = 0.01; %10^-2;
    params.A = gpamptitude;
    params.sigma = gpnoise;
    params.rbf_mu= ones(1,1)*gpbandwidth; %bandwidth of GP
    params.tol = tol;
    params.N_budget=50;
    params.epsilon_data_select=0.2;
    params.N_obstacle = 0;
    params.obs_list = [];

    RMAX = 0.0;  %we're going to subtract this from all rewards to make an equivalent mdp with prior = 0;

    %domain params
    params.gamma = 0.99;

    params.N_state_dim= 2; %2;
    params.N_act = 4;
    params.s_init = [0;0]; %[1;1]; % Start in the top left corner



    %experiment parameters
    N_eps_length =200; % Length of an episode
    countSteps=1;
   
    
    
 end
 
 firstCopy =1;

%domain params



%experiment parameters
N_eps = 200; %200 Number of episodes
N_exec = 1; %20;


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


rew_exec_ind = [];
%eval_counter = 0;

allCount = 0;
for j = 1:N_eps
    % Reset the initial state
    s_old = params.s_init;
    s_indx_old = 1;%DiscretizeState(s_old',statelist);
    [qstart,action_indx] = Q_greedy_act_mean(theta1,s_old,params,gpr1);
    disp(['episode: ',num2str(j),  '  Q0: ', num2str(qstart)]);
    step_counter = 0;
    
    for k = 1: N_eps_length
        
        %state printing
        %fprintf('state\n');
        %disp(s_old);
        
        
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
        s_new = Generic_trans(s_old,action,params);
        
        %s_indx_new = DiscretizeState(s_new',statelist);
        %Calculate The Reward
        [rew,breaker] = Generic_rew(s_old,action,params);
        rew = rew - RMAX;
        
      %  disp(['action:',num2str(action)]);
        
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
        
        
       %TJW: this was there.  Rob was scared.  act_val2 + act_val_var2 < act_val1 - act_val_var1)
            %TODO: The check needs to look at the GPs more carefully and
            %make sure the means are not going up.
            firstCopy = 0;
            %fprintf('Doing a non-stat check\n');
            theta1 = theta2;
            if(updateStyle == 1)
                fprintf('significant change for action %d!  Making the swap!!!!!!!!!!!!!!!!!!!!\n', action_indx);
                [params,gpr1,theta1] = QL_copy(params,gpr2,theta2);      
                gpr2.reinitCovar(params);
            
            elseif(updateStyle ==3 || updateStyle == 2) %KL divergence check on basis vectors      
                SIG = params.rbf_mu;                SNR = params.sigma;
                %collect the basis vectors
                
                %cs1 = gpr1.get('current_size');
                cs2 = gpr2.get('current_size');
                
                %get values from the two GPs
                for jj=1:params.N_act
                %use the correct action
                    
                    %TODO: check if not full BV (look at current store)
                    %TODO: Indexing of actions correctly
                    
                    
                    %csa1 = cs1(jj);
                    csa2 = cs2(jj);
                    %total = csa1 + csa2;
                    
                    %Checked: here is where the actions are added (*jj)
                    %B = ones(params.N_state_dim +1,total) * jj; 
                    
                    %BV1 = gpr1.getOnlyAct('BV_store',jj);
                    
                    if(isempty(cs2) || csa2 == 0)
                        continue;
                    end
                    
                    
                    actDummy = ones(1,params.N_budget) * jj;
                    BV2 = [gpr2.getOnlyAct('BV_store',jj);actDummy]; 
                    BV2 = BV2(:,1:csa2);
                    
                    
                    
                    %Checked: took care of non-full BV and correct indexing
                    %of actions
                    
                   
                    [mu_IS2, S_IS2] =  gpr2.predict(BV2,params);
                   [mu_IS1, S_IS1] =  gpr1.predict(BV2,params);
                     
                   sigBasisDiffs = (mu_IS2 + diag(S_IS2) + epsilon1 )< mu_IS1;
                   KL_DIV = sum(sigBasisDiffs);
                   
                   if(KL_DIV > 0)% && (sum(mu_diff <-0.1) > 1))
                            
                            fprintf('significant change for action %d!  Making the swap!!!!!!!!!!!!!!!!!!!!\n',jj);
                            [params,gpr1,theta1] = QL_copy(params,gpr2,theta2);
                            gpr2.reinitCovar(params);
                            for x=1:params.N_grid
                               for y=1:params.N_grid
                                    plotQ(x,y)= Q_value(theta2,[(x-1)/5;(y-1)/5],1,params,gpr2);
                                    plotPol(x,y) = 1;
                                    for a=2:params.N_act
                                        if(Q_value(theta2,[(x-1)/5.0;(y-1)/5.0],a,params,gpr2) > plotQ(x,y))
                                           plotQ(x,y)= Q_value(theta2,[(x-1)/5.0;(y-1)/5.0],a,params,gpr2);
                                            plotPol(x,y) = a;
                                        end
                                    end
                                end
                        end
                            
                            figure(fh1);
                            surf(plotQ);
                            
                            %figure(fh2);
                            %quiver(1:5,1:5, ~mod(plotPol-1,2) - ~mod(plotPol-1,2).* (-2 * (plotPol == 5)),~mod(plotPol,2)  - ~mod(plotPol,2).* (-2 * (plotPol == 4)));
                            
                            %drawnow;
                            break;
                        else
                          % fprintf('No big deal, %f  no swap.\n', KL_div);
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
   if(breaker)
    fprintf('reached goal!!!!! in %d steps\n', step_counter);
    else
    fprintf('episode cap reached\n');
    end
end


%filename = ['result_', num2str(ex), 'backup', num2str(backup), '_select', num2str(select)];  %'result_m',num2str(nm),'bw',num2str(nbw),'am',num2str(na),'nn',num2str(nn),'bg',num2str(nbg),'to',num2str(nt),'r',num2str(i)];
%save(filename,'train_Reward','train_Steps','V','diff_alpha_norm');
if(countSteps == 0)
    rews(ex,:) = train_Reward(1,:);
else
    rews(ex,:) = train_Steps(1,:);
end


end

mean_Reward = mean(rews);
std_Dev_Reward = std(rews);

filename = [params.env, 'swap'];  %'result_m',num2str(nm),'bw',num2str(nbw),'am',num2str(na),'nn',num2str(nn),'bg',num2str(nbg),'to',num2str(nt),'r',num2str(i)];
if(updateStyle == -1)
    filename = [params.env, 'no_swap'];
end
if(updateStyle == 1)
    filename = [params.env, 'aggressive_swap'];
end
if(updateStyle == 2)
    filename = [params.env, 'cautious_swap'];
end



save(filename,'mean_Reward', 'std_Dev_Reward', 'rews');

