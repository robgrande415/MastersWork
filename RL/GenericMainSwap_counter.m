
%Things to add:
%1) change the stat test to be on each individual basis point in GP2
%2) change the test to consider GP1 as no variance and GP2 to have its
%current variance
%3) change the test to just look at upper tail versus mean, be simpl
%4) change the copy and test to just iterate over each action, but you do
%need to reinit the covariance bounds on each of the actions.
%5) change the copy to instead insert the basis from gp2 into gp1 by either
%knocking out the closest one or all within a given distance


close all;
clear;
countSteps = 1;
%swapCounter = 0.1;

fh1 = figure;
%fh2 = figure;


%params.env = 'bandit';
%params.env = 'acrobot';
%params.env = 'grid';
params.env = 'puddle';


plotQ = zeros(5,5);
plotPol = zeros(5,5);
plotPolX = zeros(5,5);
plotPolY = zeros(5,5);

%old code, doesn't do anything I think
params.approximation_on = 3;
params.cl_on = 0;



if(strcmp(params.env,'puddle') == 1)
    varTol = 0.4;
    epsilon1 = 0.01;
    params.N_grid = 5;
    
    gpbandwidth = 0.1;
    gpamptitude = 1;%0.5; %NOTE: This parameter is controlling the initial variance estimate %1
    gpnoise = sqrt(0.5); %10^-1;
    tol  = 1e-10;%0.01;  %0001; %10^-2;
    params.A = gpamptitude;
    params.sigma = gpnoise;
    params.rbf_mu= ones(1,1)*gpbandwidth; %bandwidth of GP
    params.tol = tol;
    params.N_budget=300; %200;
    params.epsilon_data_select=0.2;
    params.N_obstacle = 0;
    params.obs_list = [];

    RMAX = 0.0;  %we're going to subtract this from all rewards to make an equivalent mdp with prior = 0;

    %domain params
    params.gamma = 0.8;

    params.N_state_dim= 2; 
    params.N_act = 4;
    params.s_init = [0;0]; %[1;1]; % Start in the top left corner



    %experiment parameters
    N_eps_length =200; % Length of an episode
    countSteps=1;    
    
 end
 


%experiment parameters
N_eps = 200; % Number of episodes
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
     
 
    gpr1 = onlineGP.onlineGP_MA(params.N_act,params.rbf_mu,params.sigma,params.N_budget,params.tol,params.A);


rew_exec_ind = [];
%eval_counter = 0;

allCount = 0;

for j = 1:N_eps
    
    step_counter = 0;
    s_old = [params.s_init];
    [qstart,action] = gpr1.getMax(s_old); 
    disp(['episode: ',num2str(j),  '  Q0: ', num2str(qstart)]);
    swapMark = 0;
    
    
    for k = 1: N_eps_length
        
        allCount = allCount + 1;
        %Greed is good.	
        step_counter = step_counter + 1;
        
        %get next action
        [Q_opt,action] = gpr1.getMax(s_old);
        
        % Next State
        s_new = [Generic_trans(s_old(1:params.N_state_dim,1),action,params)];
        
        %s_indx_new = DiscretizeState(s_new',statelist);
        %Calculate The Reward
        [rew,breaker] = Generic_rew(s_old(1:params.N_state_dim,1),action,params);
        rew = rew - RMAX;
        
      
        train_Reward(1,j) = train_Reward(1,j) + params.gamma^(k-1)*rew;
        train_Steps(1,j) = k;
        
        y = rew;
        if(~breaker)
            y = y+ params.gamma * gpr1.getMax(s_new);
        end
        
        if(y > 0)
            y
        end
        
        gpr1.update(s_old,action,y);
        
        
        [yhat, var] = gpr1.predict(s_old,action,'full');
        if(var < varTol && (yhat - y) < epsilon1) %HELP: this needs to be fixed to instead be a bunch of values that are way off in this vicinity. 
            %now we have a poin that is old and whose value is off
            swapMark = 1;
            
            %fprintf('Doing a basis elimination!!!\n');
            
            %knock out nearest basis vector
            %gpr1.removeClosestBasis(s_old,action);
            
            %reinitialize with old predicted value
            %gpr1.update(s_old,action,yhat);
        end
        
           
	% if reachs the goal breaks the episode
        if breaker
            break;
        end
        
        s_old = s_new;
        
        
    end
    
    if(swapMark > 0)
            fprintf('knocked out some basis points\n');
    end
    
    %plot
    figure(1);
    [XX YY] = meshgrid(linspace(0,1,20));
      X = XX(:);
      Y = YY(:);
    nmodels = gpr1.n_Acts;
    for iii=1:1
        %subplot(1,nmodels,iii)
      %gpr_temp = gpr1.getGP(iii);  
      p_color = [(iii/nmodels) 1 1];  
      x = [X';Y'];
      [mean_post] = gpr1.getMax(x);
      mean_post = reshape(mean_post,20,20);
      
      surf(XX,YY,mean_post)
      pause(0.00000001);
    end    
    
    %alpha = gpr.get('alpha_store');
    %if j>1
    %    diff_alpha_norm(j-1) = norm(alpha-alpha_old,'fro');
    %end
    %alpha_old = alpha;
   if(breaker)
    fprintf('reached goal!!!!! in %d steps\n', step_counter);
   else
       fprintf('episode cap reached...\n');
       %if(swap_counter == 0)
       %    s_old
       %end
    %fprintf('episode cap reached at [%d  %d] with values %f % f\n', s_old(1), s_old(2), gpr1.getMax(s_old), gpr2.getMax(s_old));
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
