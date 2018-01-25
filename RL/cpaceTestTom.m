
%Things to add:
%1) change the stat test to be on each individual basis point in GP2
%2) change the test to consider GP1 as no variance and GP2 to have its
%current variance
%3) change the test to just look at upper tail versus mean, be simpl
%4) change the copy and test to just iterate over each action, but you do
%need to reinit the covariance bounds on each of the actions.
%5) change the copy to instead insert the basis from gp2 into gp1 by either
%knocking out the closest one or all within a given distance

clear;
close all;
%clear;
countSteps = 1;

fh1 = figure;
%fh2 = figure;


%params.env = 'bandit';
%params.env = 'acrobot';
%params.env = 'grid';
params.env = 'puddle';
params.env = 'cartpole';
%params.env = 'glider';
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
    epsilon1 = 0.1;
    if(updateStyle == 2)
        epsilon1 = 0.2; %threshold basis points needs to come down by before we do a check
    end

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

    N_eps_length =2; % Length of an episode

 elseif(strcmp(params.env,'acrobot')==1)

     %addpath('C:\cygwin\home\Tom\mit\gpq\GPQlearning\acrobot');
     epsilon1 = 0.01;
    if(updateStyle == 2)
        epsilon1 = 1; %threshold basis points needs to come down by before we do a check
    end

%GP params'



    gpbandwidth = 0.25;
    gpamptitude = 1.0; %NOTE: This parameter is controlling the initial variance estimate %1
    gpnoise = sqrt(0.01); 
    gptol  = 0.01; %10^-2;
    params.A = gpamptitude;
    params.sigma = gpnoise;
    params.rbf_mu= ones(1,1)*gpbandwidth; %bandwidth of GP
    params.tol = gptol;
    params.N_budget=200;
    %params.epsilon_data_select=0.2;
    RMAX = 0.0;  %we're going to subtract this from all rewards to make an equivalent mdp with prior = 0;

    params.gamma = 0.97; %0.95   
    %params.N_state = 1;
    %N_state = params.N_state;
    %params.statelist = [1];
    params.N_state_dim= 4; %2;
    params.N_act = 3;
    params.s_init = [0;0;0;0]; %[1;1]; % Start in the top left corner
    params.a_init = 2;

    N_eps_length =250; % Length of an episode
    countSteps = 1;
    
 elseif(strcmp(params.env,'cartpole') == 1)
     
     %addpath('C:\cygwin\home\Tom\mit\gpq\GPQlearning\acrobot');
         epsilon1 = 0.001;
        if(updateStyle == 2)
            epsilon1 = 1; %threshold basis points needs to come down by before we do a check
        end

    %GP params'


        SIGMA_TOL = 1/8;
        gpbandwidth = 0.15;
        gpamptitude = 1; %NOTE: This parameter is controlling the initial variance estimate %1
        gpnoise = sqrt(0.1); 
        gptol  = 0.05; %10^-2;
        params.A = gpamptitude;
        params.sigma = gpnoise;
        params.rbf_mu= ones(1,1)*gpbandwidth; %bandwidth of GP
        params.tol = gptol;
        params.N_budget=200;
        %params.epsilon_data_select=0.2;
        RMAX = 0.0;  %we're going to subtract this from all rewards to make an equivalent mdp with prior = 0;

        params.gamma = 0.99; %0.95   
        %params.N_state = 1;
        %N_state = params.N_state;
        %params.statelist = [1];
        params.N_state_dim= 4; %2;
        params.N_act = 3;
        params.s_init = [0;0;0;0]; %[1;1]; % Start in the top left corner
        params.a_init = 1;

        
        params.rew_bw = 2; %bandwidth of GP rew
        params.trans_bw = 2; %bandwidth of GP rew
        params.Q_bw = 0.5;
        
        N_eps_length =500; % Length of an episode
        countSteps = 0;
        params.Lip = pi; %1.9 for swing, 2 for balance 
        params.paramVec = [0.5 0.2 0.006 0.3 9.81 0.2]; %M m I l g b

         %CPD stuff
        params.bin_size = 5;
        params.KL_tol = inf;
        params.bin_tol = inf;


        optimism = sqrt(0.5^2 * log(2/0.05) / (2 * params.sigma^2)); % Vm^2 log(2/delta) / 2 omega^2
        RMAX_Eps_Thresh =0.7;
        params.epsilon_conv = 0.01;
        
 elseif(strcmp(params.env,'grid') == 1)
    epsilon1 = 0.01;
    if(updateStyle == 2) %cautious
        epsilon1 = 0.5; %threshold basis points needs to come down by before we do a check
    end

    
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
    epsilon1 = 0.05;
    params.N_grid = 5;
    if(updateStyle == 2) %cautious
        epsilon1 = 0.5; %threshold basis points needs to come down by before we do a check
    end

    
    gpbandwidth = 0.15;
    gpamptitude = 0.5; %NOTE: This parameter is controlling the initial variance estimate %1
    gpnoise = sqrt(0.1); %10^-1;
    tol  = 0.001;  %0001; %10^-2;
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
    params.gamma = 0.99;

    params.N_state_dim= 2; %2;
    params.N_act = 4;
    params.s_init = [0;0]; %[1;1]; % Start in the top left corner



    %experiment parameters
    N_eps_length =100; % Length of an episode
    countSteps=1;
   
    
    elseif(strcmp(params.env,'glider') == 1)
    epsilon1 = 0.001;
    if(updateStyle == 2) %cautious
        epsilon1 = 0.5; %threshold basis points needs to come down by before we do a check
    end

    
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
SNR = gpnoise;


%experiment parameters
N_eps = 200; % Number of episodes
N_exec = 3; %20;
plotme = 0;

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
    
     lip = 9;% *2; % 18/sqrt(2*0.7);
    cpaceq = instanceBased.Cpace(params.N_act,lip,100000,3,0,params.gamma,params.N_state_dim, 0.05);
    
   
    timer = zeros(N_exec,N_eps * N_eps_length);
    

    rew_exec_ind = [];
    %eval_counter = 0;

    allCount = 0;

    %init actions with 0
    %{
    for jj = 1:4
        gpr1.update([-0.2;-0.2],jj,0);
        gpr1.update([1.2;-0.2],jj,0);
        gpr1.update([1.2;1.2],jj,0);
        gpr1.update([-0.2;1.2],jj,0);

        gpr2.update([-0.2;-0.2],jj,0);
        gpr2.update([1.2;-0.2],jj,0);
        gpr2.update([1.2;1.2],jj,0);
        gpr2.update([-0.2;1.2],jj,0);
    end
    %}
    for j = 1:N_eps
        %if mod(j,50) == 0
        %    epsilon1 = epsilon1/2;
        %end
        % Reset the initial state
        s_old = params.s_init;
        [qstart,action_indx] = cpaceq.getMax(s_old); 
        disp(['run: ', num2str(ex)  ,' episode: ',num2str(j),  '  Q0: ', num2str(qstart)]);
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

            startTime = cputime;
            
            %not using the variance here, because things will be set
            %optimistically already
            [Q_opt,action_indx] = cpaceq.getMax(s_old);
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

            
            if(~breaker)
                if(k == N_eps_length)
                    cpaceq.update(s_old,action_indx,rew,s_new,2);
                else
                    cpaceq.update(s_old,action_indx,rew,s_new,0);
                end
            else
                cpaceq.update(s_old,action_indx,rew,[],1);
                endTime = cputime;
                timer(ex,allCount) = (endTime - startTime);
                break;
            end

            s_old = s_new;
            endTime = cputime;
            timer(ex,allCount) = (endTime - startTime);
            %fprintf('%f\n', timer(1,allCount));

        end

        
        %plot
        if plotme 
            fprintf('plotting \n');        %plot
            figure(1);
            [XX YY] = meshgrid(linspace(0,1,21));
              X = XX(:);
              Y = YY(:);
                %subplot(1,nmodels,iii)
              %gpr_temp = gpr1.getGP(iii);  
              mean_post = zeros(21,21);
              for x=1:21
                  for y=1:21
                    mean_post(x,y) = cpaceq.getMax([(x-1) * .05;(y-1) * .05]);
                  end
              end

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
           fprintf('episode cap reached, chose action %d\n', action);
           %if(swap_counter == 0)
               s_old
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

filename = [params.env, '_cpace'];  %'result_m',num2str(nm),'bw',num2str(nbw),'am',num2str(na),'nn',num2str(nn),'bg',num2str(nbg),'to',num2str(nt),'r',num2str(i)];

save(filename,'mean_Reward', 'std_Dev_Reward', 'rews', 'timer');
