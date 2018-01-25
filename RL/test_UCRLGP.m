
%Things to add:
%1) change the stat test to be on each individual basis point in GP2
%2) change the test to consider GP1 as no variance and GP2 to have its
%current variance
%3) change the test to just look at upper tail versus mean, be simpl
%4) change the copy and test to just iterate over each action, but you do
%need to reinit the covariance bounds on each of the actions.
%5) change the copy to instead insert the basis from gp2 into gp1 by either
%knocking out the closest one or all within a given distance

%clear;
close all;
%clear;
countSteps = 1;
addpath Cart_sim/
addpath F16/
global f16_actions
global A_f_16 B_f_16


%params.env = 'bandit';
%params.env = 'acrobot';
%params.env = 'grid';
params.env = 'puddle';
%params.env = 'cartpole';
%params.env = 'LTI';
%params.env = 'glider';
%params.env = 'f16';
%params.env = '747';
%variations for doing the updates:
updateStyle = 3; 
%-1 = none, 1 = aggressive, 2 = cautious, 3 = swap

plotQ = zeros(5,5);
plotPol = zeros(5,5);
plotPolX = zeros(5,5);
plotPolY = zeros(5,5);

switch params.env
    case 'bandit' 
        
        epsilon1 = 0.05;
        if(updateStyle == 2)
            epsilon1 = 0.2; %threshold basis points needs to come down by before we do a check
        end
        params.Lip = 1;
        SIGMA_TOL = 0.2;
    %GP params

        gpbandwidth = 0.5;
        gpamptitude = 1; %NOTE: This parameter is controlling the initial variance estimate %1
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
        countSteps = 0;
    case 'acrobot'

         %addpath('C:\cygwin\home\Tom\mit\gpq\GPQlearning\acrobot');
         epsilon1 = 0.001;
        if(updateStyle == 2)
            epsilon1 = 1; %threshold basis points needs to come down by before we do a check
        end

    %GP params'


        SIGMA_TOL = 1/2;
        gpbandwidth = 0.15;
        gpamptitude = 1; %NOTE: This parameter is controlling the initial variance estimate %1
        gpnoise = sqrt(0.1); 
        gptol  = 0.01; %10^-2;
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
        params.a_init = 2;

        N_eps_length =500; % Length of an episode
        countSteps = 1;
        params.Lip = 7.5;
    
    
    
    
    
    case 'cartpole'

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
        gptol  = 0.01; %10^-2;
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

        N_eps_length =500; % Length of an episode
        countSteps = 0;
        params.Lip = pi; %1.9 for swing, 2 for balance 
        params.paramVec = [0.5 0.2 0.006 0.3 9.81 0.2]; %M m I l g b


    case 'grid'
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
        params.Lip = 80;


        %experiment parameters
        N_eps_length =200; % Length of an episode
        
    case 'puddle'
        epsilon1 = 0.001;
        params.N_grid = 5;
        if(updateStyle == 2) %cautious
            epsilon1 = 0.5; %threshold basis points needs to come down by before we do a check
        end

        SIGMA_TOL = 0.05;
        gpbandwidth = 0.15; %0.15;
        gpamptitude = 1; %NOTE: This parameter is controlling the initial variance estimate %1
        gpnoise = sqrt(0.1); %10^-1;
        tol  = 0.05;  %0001; %10^-2;
        params.A = gpamptitude;
        params.sigma = gpnoise;
        params.rew_bw = 0.25; %bandwidth of GP rew
        params.trans_bw = 0.25; %bandwidth of GP rew
        params.Q_bw = 0.15; %bandwidth of GP rew
        params.tol = tol;
        params.N_budget=100; %200;
        params.epsilon_data_select=0.2;
        params.N_obstacle = 0;
        params.obs_list = [];
        params.Lip = 18/sqrt(2) * 1; %2-norm
        params.Lip = 10; %1-norm
        RMAX = 0.0;  %we're going to subtract this from all rewards to make an equivalent mdp with prior = 0;

        %domain params
        params.gamma = 1;

        params.N_state_dim= 2; %2;
        params.N_act = 4;
        params.s_init = [0;0]; %[1;1]; % Start in the top left corner



        %experiment parameters
        N_eps_length = 200; % Length of an episode
        countSteps=1;

    case 'LTI'
        epsilon1 = 0.05;
        if(updateStyle == 2) %cautious
            epsilon1 = 0.5; %threshold basis points needs to come down by before we do a check
        end

        SIGMA_TOL = 0.1;
        gpbandwidth = 0.03; %0.15;
        gpamptitude = 1; %NOTE: This parameter is controlling the initial variance estimate %1
        gpnoise = sqrt(0.1); %10^-1;
        tol  = 0.001;  %0001; %10^-2;
        params.A = gpamptitude;
        params.sigma = gpnoise;
        params.rbf_mu= ones(1,1)*gpbandwidth; %bandwidth of GP
        params.tol = tol;
        params.N_budget=inf; %200;
        params.epsilon_data_select=0.2;
        params.N_obstacle = 0;
        params.obs_list = [];
        %params.Lip = 400;
        RMAX = 0.0;  %we're going to subtract this from all rewards to make an equivalent mdp with prior = 0;
        params.Lip = 15; %20 worked for randn0.01, 30 for 0.02
 
        %domain params
        params.gamma = 0.99;

        params.N_state_dim= 3; %2;
        params.N_act = 3;
        params.s_init = [0;0;0]; %[1;1]; % Start in the top left corner



        %experiment parameters
        N_eps_length = 200; % Length of an episode
        countSteps=1;
    
    case 'glider'
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
    case '747'

         %addpath('C:\cygwin\home\Tom\mit\gpq\GPQlearning\acrobot');
         epsilon1 = 0.01;
        if(updateStyle == 2)
            epsilon1 = 1; %threshold basis points needs to come down by before we do a check
        end

    %GP params'


        SIGMA_TOL = 1/2;
        gpbandwidth = 0.25;
        gpamptitude = 0.5; %NOTE: This parameter is controlling the initial variance estimate %1
        gpnoise = sqrt(0.1); 
        gptol  = 0.01; %10^-2;
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
        params.N_state_dim= 5; %2;
        params.N_act = 9;
        params.s_init = [0;0;0;0;0]; %[1;1]; % Start in the top left corner
        params.a_init = 1;

        N_eps_length = 300; % Length of an episode
        countSteps = 0;
        params.Lip = 10; %1.9 for swing, 2 for balance 

    case 'f16'

         %addpath('C:\cygwin\home\Tom\mit\gpq\GPQlearning\acrobot');
         epsilon1 = 0.05; %0.05 worked

        if(updateStyle == 2)
            epsilon1 = 1; %threshold basis points needs to come down by before we do a check
        end

    %GP params'


        SIGMA_TOL = 0.1;
        gpbandwidth = 0.05;
        gpamptitude = 1; %NOTE: This parameter is controlling the initial variance estimate %1
        gpnoise = sqrt(0.1); 
        gptol  = 0.01; %10^-2;
        params.A = gpamptitude;
        params.sigma = gpnoise;
        params.rbf_mu= ones(1,1)*gpbandwidth; %bandwidth of GP
        params.tol = gptol;
        params.N_budget=400;
        %params.epsilon_data_select=0.2;
        RMAX = 0.0;  %we're going to subtract this from all rewards to make an equivalent mdp with prior = 0;

        params.gamma = 0.99; %0.95   
        %params.N_state = 1;
        %N_state = params.N_state;
        %params.statelist = [1];
        params.N_state_dim= 6; %2;
        params.N_act = 3;
        params.s_init = zeros(6,1); %[1;1]; % Start in the top left corner
        params.a_init = 1;
        params.s_init(5) = -0; %times 100
        N_eps_length = 600; % Length of an episode
        countSteps = 0;
        params.Lip = 10; %5 worked
        %linear model
        load f_16_dynamics
        f16_actions = [zeros(1,7); 0 -1 1 -5 5 -10 10]*1;
        %nonlinear model
        %{
        f16_init;
        u0
        f16_actions = repmat(u0,1,9);
        f16_actions(1,1:3) = u0(1)*1.05;
        f16_actions(1,7:9) = u0(1)*0.95;
        f16_actions(3,1:3:9) = u0(3)-2.5/180*pi;
        f16_actions(3,3:3:9) = u0(3)+2.5/180*pi;
        %}
 end
 
 firstCopy =1;

%domain params
SNR = gpnoise;
BV_old = 1;
BV = 1;
alpha1 = 0;
alpha2 = 0;
alpha3 = 0;
alpha4 = 0;


%experiment parameters
N_eps = 40; % Number of episodes
N_exec = 1; %20;
plotme = 1;
multi_task = 0;
animateme=0;
animate_freq = 50;
rand('state',1000*sum(clock));
avgr_on = 1;

rews = zeros(N_exec,N_eps);   
traj = zeros(params.N_state_dim,N_eps_length);
act_trace = zeros(1,N_eps_length);

for ex=1:N_exec

    train_Reward = zeros(1,N_eps);
    train_Steps = zeros(1,N_eps);
     
 
    gpr_rew = MBRL.onlineGP_MA_Scalar(params.N_act,params.rew_bw,params.sigma,params.N_budget,params.tol,params.A);
    gpr_rew_old = MBRL.onlineGP_MA_Scalar(params.N_act,params.rew_bw,params.sigma,params.N_budget,params.tol,params.A);
    gpr_rew_stack.gp{1} = MBRL.onlineGP_MA_Scalar(params.N_act,params.rew_bw,params.sigma,params.N_budget,params.tol,params.A);
    gpr_rew_stack.gp{2} = MBRL.onlineGP_MA_Scalar(params.N_act,params.rew_bw,params.sigma,params.N_budget,params.tol,params.A);
    gpr_rew_stack.gp{3} = MBRL.onlineGP_MA_Scalar(params.N_act,params.rew_bw,params.sigma,params.N_budget,params.tol,params.A);
    gpr_rew_stack.gp{4} = MBRL.onlineGP_MA_Scalar(params.N_act,params.rew_bw,params.sigma,params.N_budget,params.tol,params.A);

    
    gpr_empty = MBRL.onlineGP_MA_Scalar(params.N_act,params.rbf_mu,params.sigma,params.N_budget,params.tol,params.A);
    gpr_Q = MBRL.onlineGP_MA_Scalar(params.N_act,params.Q_bw,params.sigma,params.N_budget,params.tol,params.A);
    gpr_Q_temp = MBRL.onlineGP_MA_Scalar(params.N_act,params.Q_bw,params.sigma,params.N_budget,params.tol,params.A);
    gpr_Q_stack.gp{1} = MBRL.onlineGP_MA_Scalar(params.N_act,params.Q_bw,params.sigma,params.N_budget,params.tol,params.A);
    gpr_Q_stack.gp{2} = MBRL.onlineGP_MA_Scalar(params.N_act,params.Q_bw,params.sigma,params.N_budget,params.tol,params.A);
    gpr_Q_stack.gp{3} = MBRL.onlineGP_MA_Scalar(params.N_act,params.Q_bw,params.sigma,params.N_budget,params.tol,params.A);
    gpr_Q_stack.gp{4} = MBRL.onlineGP_MA_Scalar(params.N_act,params.Q_bw,params.sigma,params.N_budget,params.tol,params.A);

    
    
    gpr_trans = MBRL.onlineGP_MA_Vector(params.N_act,params.trans_bw,params.sigma,params.N_budget,params.tol,params.N_state_dim);
    gpr_trans_stack.gp{1} = MBRL.onlineGP_MA_Vector(params.N_act,params.trans_bw,params.sigma,params.N_budget,params.tol,params.N_state_dim);
    gpr_trans_stack.gp{2} = MBRL.onlineGP_MA_Vector(params.N_act,params.trans_bw,params.sigma,params.N_budget,params.tol,params.N_state_dim);
    gpr_trans_stack.gp{3} = MBRL.onlineGP_MA_Vector(params.N_act,params.trans_bw,params.sigma,params.N_budget,params.tol,params.N_state_dim);
    gpr_trans_stack.gp{4} = MBRL.onlineGP_MA_Vector(params.N_act,params.trans_bw,params.sigma,params.N_budget,params.tol,params.N_state_dim);

    %averager stuff
    timer = zeros(N_exec,N_eps * N_eps_length);
    
    if(updateStyle == -1)
        gpr1 = gpr2;
    end


    rew_exec_ind = [];
    %eval_counter = 0;

    allCount = 0;

    
    for j = 1:N_eps
        
        % Reset the initial state
        s_old = params.s_init;
        [qstart,action] = gpr_Q.getMax(s_old);
        
        %select MDP
        if multi_task
            mdp_num = mod(ceil(j/10)-1,4)+1;
            %if mdp switches, save old model and reinit stuff
            if mdp_num ~= mod(ceil((j-1)/10)-1,4)+1
                if mdp_num > 1
                    gpr_Q_stack.gp{mdp_num-1}.copy(gpr_Q);
                    gpr_trans_stack.gp{mdp_num-1}.copy(gpr_trans);
                    gpr_rew_stack.gp{mdp_num-1}.copy(gpr_rew);
                    gpr_Q.reinitAll();
                    gpr_Q_temp.reinitAll();
                    gpr_trans.reinitAll();
                    gpr_rew.reinitAll();
                end
            end
        else
            mdp_num = 1;
        end
        disp(['episode: ',num2str(j),  '  Q0: ', num2str(qstart), '  MDP: ', num2str(mdp_num)]);
        step_counter = 0;
        swap_counter  = 0;

        for k = 1: N_eps_length
            
            
            % Increment the step counter
            allCount = allCount + 1;
            step_counter = step_counter + 1;
            
            %timer
            %startTime = cputime;
            tic;       
            
            
            %act greedily
            [Q_opt,action] = gpr_Q.getMax(s_old);
            
            % Next State
            s_new = Generic_trans(s_old,action,params);
            
            %save trajectory on episode 100
            if mod(j,animate_freq) == 0
                traj(:,k) = s_old;
                act_trace(1,k) = action;
            end
            
            %Calculate The Reward
            [rew,breaker] = Generic_rew(s_old,action,params,mdp_num);
            rew = rew - RMAX;
            
            %update reward model
            gpr_rew.update(s_old,action,rew);
            
            %update dynamics model
            diff = s_new -s_old;
            gpr_trans.update(s_old,action,diff);
            
            
            
            if mod(k,10) == 0 %breaker || k==200
                %run bellman update
                
                %gpr_Q_temp.reinitAll();
                gpr_Q_temp.copy(gpr_Q);
                gpr_Q.reinitAll();
                for bell = 1:1
                    BV = [];
                    for act = 1:params.N_act
                        BV = [BV,gpr_rew.getGP(act).get('BV')];
                    end
                    BV = unique(BV','rows')';
                    
                    
                    %2
                    for bv = 1:size(BV,2)
                        for act = 1:params.N_act
                            [q_new, v_rew] = gpr_rew.predict(BV(:,bv),act);
                            diff_new = gpr_trans.predict(BV(:,bv),act);
                            
                            %GP-RMAX optimism 
                            %{
                            if v_rew > SIGMA_TOL
                                gpr_Q.update(BV(:,bv),act,0);
                                continue;
                            end
                            %}
                            %confidence bound
                            
                            optimism = sqrt(1^2 * log(2/0.05) / (2 * params.sigma^2)); % Vm^2 log(2/delta) / 2 omega^2
                            q_new = q_new + 2*sqrt(v_rew);
                            q_new(q_new>0) = 0; %max out
                            
                            
                            state_new = BV(:,bv) + diff_new;
                            Q_next = gpr_Q_temp.getMax(state_new);
                            q_new = q_new + params.gamma * Q_next;
                            gpr_Q.update(BV(:,bv),act,q_new);
                        end
                    end
                    
                    
                    gpr_Q_temp.copy(gpr_Q);
                    gpr_Q.reinitAll();
                end
                if 1
                            %plot
                    figure(1);
                    [XX YY] = meshgrid(linspace(0,1,21));
                      X = XX(:);
                      Y = YY(:);
                        %subplot(1,nmodels,iii)
                      %gpr_temp = gpr1.getGP(iii);  
                      x = [X';Y'];

                      [mean_post] = gpr_Q_temp.getMax(x);

                      mean_post = reshape(mean_post,21,21);

                      surf(XX,YY,mean_post)
                      axis([0 1 0 1 -4 1])
                      pause(0.001);

                end
            end
            gpr_Q.copy(gpr_Q_temp);
            %record to stack
            train_Reward(1,j) = train_Reward(1,j) + params.gamma^(k-1)*rew;
            train_Steps(1,j) = k;
            
            BV_old = BV;
            gpr_rew_old.copy(gpr_rew);

            % if reachs the goal breaks the episode
            if breaker
                %update reward model
                gpr_rew.update(s_new,action,0);
                break;
            end

            s_old = s_new;
            %endTime = cputime;
            endTime = toc;
            %timer(ex,allCount) = (endTime - startTime);
            timer(ex,allCount) = endTime ;

        end

        

        %animate
        if animateme && mod(j,animate_freq) == 0            
            traj = traj';
            trim = find(act_trace == 0,1);
            if isempty(trim)
                trim = length(act_trace);
            end
            
            switch params.env
                case 'cartpole'
                    animate_cart(1:trim,traj);
                case 'acrobot'
                    animate_acrobot(1:trim,traj,1,1,0);
                case 'f16'
                    animate_airplane(1:trim,traj,act_trace);
                case 'LTI'
                    animate_LTI(1:trim,traj,act_trace);
            end
        end
        
        traj = zeros(params.N_state_dim,N_eps_length);
        act_trace = zeros(1,N_eps_length);
           
        %plot
        if plotme 
                    %plot
            figure(1);
            [XX YY] = meshgrid(linspace(0,1,21));
              X = XX(:);
              Y = YY(:);
                %subplot(1,nmodels,iii)
              %gpr_temp = gpr1.getGP(iii);  
              x = [X';Y'];
              %{
               K1 = MBRL.kernel(x,BV,params.rbf_mu,params.A);
              mu1 = K1 * alpha1;
              mu2 = K1 * alpha2;
              mu3 = K1 * alpha3;
              mu4 = K1 * alpha4;
              mu_stack = [mu1, mu2, mu3, mu4];
              mean_post = max(mu_stack')';
              %}
              [mean_post] = gpr_Q.getMax(x);
              %[mean_post] = gpr2.getMax(x); 
              mean_post = reshape(mean_post,21,21);

              surf(XX,YY,mean_post)
              pause(0.00000001);
              
        end

       if(breaker)
            fprintf('reached goal!!!!! in %d steps\n', step_counter);
       else
           fprintf('episode cap reached, chose action %d', action);
       end
        
    end


    %filename = ['result_', num2str(ex), 'backup', num2str(backup), '_select', num2str(select)];  %'result_m',num2str(nm),'bw',num2str(nbw),'am',num2str(na),'nn',num2str(nn),'bg',num2str(nbg),'to',num2str(nt),'r',num2str(i)];
    %save(filename,'train_Reward','train_Steps','V','diff_alpha_norm');
    if(countSteps == 0)
        rews(ex,:) = train_Reward(1,:);
    else
        rews(ex,:) = train_Steps(1,:);
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



save(filename,'mean_Reward', 'std_Dev_Reward', 'rews', 'timer');




end

%save last mdp
gpr_Q_stack.gp{mdp_num}.copy(gpr_Q);
gpr_trans_stack.gp{mdp_num}.copy(gpr_trans);
gpr_rew_stack.gp{mdp_num}.copy(gpr_rew);

