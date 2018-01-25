
%Things to add:
%1) change the stat test to be on each individual basis point in GP2
%2) change the test to consider GP1 as no variance and GP2 to have its
%current variance
%3) change the test to just look at upper tail versus mean, be simple
%4) change the copy and test to just iterate over each action, but you do
%need to reinit the covariance bounds on each of the actions.
%5) change the copy to instead insert the basis from gp2 into gp1 by either
%knocking out the closest one or all within a given distance


close all;
clear;
countSteps = 1;

%fh2 = figure;


%params.env = 'bandit';
%params.env = 'acrobot';
%params.env = 'grid';
params.env = 'puddle';

%variations for doing the updates:



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
    
%GP params

    gpbandwidth = 0.5;
    gpamptitude = 0.5; %NOTE: This parameter is controlling the initial variance estimate %1
    gpnoise = sqrt(0.1); 
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
    
%GP params'



    gpbandwidth = 0.5;
    gpamptitude = 1.0; %NOTE: This parameter is controlling the initial variance estimate %1
    gpnoise = 0.01; 
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
   
    
    gpbandwidth = 0.5;
    gpamptitude = 0.5;%0.5; %NOTE: This parameter is controlling the initial variance estimate %1
    gpnoise = sqrt(0.1);%sqrt(.1); %10^-1;
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
    
    params.N_grid = 5;
    
    
    gpbandwidth = 0.1;
    gpamptitude = 0.5; %NOTE: This parameter is controlling the initial variance estimate %1
    gpnoise = sqrt(0.1); %10^-1;
    tol  = 0.01;  %0001; %10^-2;
    params.A = gpamptitude;
    params.sigma = gpnoise;
    params.rbf_mu= ones(1,1)*gpbandwidth; %bandwidth of GP
    params.tol = tol;
    params.N_budget=200; %200;
    params.epsilon_data_select=0.2;
    params.N_obstacle = 0;
    params.obs_list = [];

    RMAX = 0.0;  %we're going to subtract this from all rewards to make an equivalent mdp with prior = 0;

    %domain params
    params.gamma = 0.9;

    params.N_state_dim= 2; %2;
    params.N_act = 4;
    params.s_init = [0;0]; %[1;1]; % Start in the top left corner



    %experiment parameters
    N_eps_length =200; % Length of an episode
    countSteps=1;
   
    
    elseif(strcmp(params.env,'glider') == 1)
   
    
    gpbandwidth = 0.25;
    gpamptitude = 0.5; %NOTE: This parameter is controlling the initial variance estimate %1
    gpnoise = 0.1; %10^-1;
    tol  = 0.00001; %10^-2;
    params.A = gpamptitude;
    params.sigma = gpnoise;
    params.rbf_mu= ones(1,1)*gpbandwidth; %bandwidth of GP
    params.tol = tol;
    params.N_budget=200;
    params.epsilon_data_select=0.2;
    params.N_obstacle = 0;
    params.obs_list = [];
    params.origSprime = [];
    params.origSi = [];
    params.s_prime = [];
    params.s_primen = [];
    params.si = [];
    params.si_n = [];
    
    
    
    RMAX = 0.0;  %we're going to subtract this from all rewards to make an equivalent mdp with prior = 0;

    %domain params
    params.gamma = 0.99;

    params.N_state_dim= 2; %2;
    params.N_act = 3;
    params.s_init = [0;0]; %[1;1]; % Start in the top left corner



    %experiment parameters
    N_eps_length =200; % Length of an episode
    countSteps=1;
    
    
 end
 
 firstCopy =1;

%domain params



%experiment parameters
N_eps = 200; % Number of episodes
N_exec = 1; %20;


rand('state',1000*sum(clock));


rews = zeros(N_exec,N_eps);   
        
for ex=1:N_exec

   
    train_Reward = zeros(1,N_eps);
     train_Steps = zeros(1,N_eps);
     
 
    gpr1 = onlineGP.onlineGP_MA(params.N_act,params.rbf_mu,params.sigma,params.N_budget,params.tol,params.A);
    

    rew_exec_ind = [];


    allCount = 0;

    for j = 1:N_eps
    
        % Reset the initial state
        s_old = params.s_init;

        [qstart,action_indx] = gpr1.getMax(s_old); 
        disp(['episode: ',num2str(j),  '  Q0: ', num2str(qstart)]);
        step_counter = 0;
        swap_counter  = 0;

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
            [Q_opt,action_indx] = gpr1.getMax(s_old);
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

            y = rew;
            if(~breaker)
                y = y+ params.gamma * gpr1.getMax(s_new);
            end

            gpr1.update(s_old,action_indx,y);


        % if reachs the goal breaks the episode
            if breaker
                break;
            end

            s_old = s_new;


        end
    
    
    %plot
    
    %alpha = gpr.get('alpha_store');
    %if j>1
    %    diff_alpha_norm(j-1) = norm(alpha-alpha_old,'fro');
    %end
    %alpha_old = alpha;
       if(breaker)
            fprintf('reached goal!!!!! in %d steps\n', step_counter);
       else
           fprintf('episode cap reached\n');
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


filename = [params.env, '_GPQ'];


save(filename,'mean_Reward', 'std_Dev_Reward', 'rews');
