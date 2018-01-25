
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
%params.env = 'puddle';
%params.env = 'cartpole';
%params.env = 'LTI';
%params.env = 'glider';
params.env = 'f16';
%params.env = '747';
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


switch params.env
    case 'bandit' 
        
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

        N_eps_length =1; % Length of an episode

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

        SIGMA_TOL = 1/2;
        gpbandwidth = 0.05; %0.15;
        gpamptitude = 0.5; %NOTE: This parameter is controlling the initial variance estimate %1
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
        params.Lip = 18/sqrt(2) * 1; %2-norm
        params.Lip = 9; %1-norm
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
        epsilon1 = 0.0001;
        if(updateStyle == 2) %cautious
            epsilon1 = 0.5; %threshold basis points needs to come down by before we do a check
        end

        SIGMA_TOL = 1/2;
        gpbandwidth = 0.15; %0.15;
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
        params.Lip = 10;

        %domain params
        params.gamma = 0.99;

        params.N_state_dim= 2; %2;
        params.N_act = 3;
        params.s_init = [1;0]; %[1;1]; % Start in the top left corner



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
         epsilon1 = 0.01;
        if(updateStyle == 2)
            epsilon1 = 1; %threshold basis points needs to come down by before we do a check
        end

    %GP params'


        SIGMA_TOL = 1/8;
        gpbandwidth = 0.5;
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
        params.N_state_dim= 6; %2;
        params.N_act = 3;
        params.s_init = zeros(6,1); %[1;1]; % Start in the top left corner
        params.a_init = 1;
        params.s_init(5) = 0; %times 100
        N_eps_length = 600; % Length of an episode
        countSteps = 0;
        params.Lip = 13; %1.9 for swing, 2 for balance 
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


%experiment parameters
N_eps = 10000; % Number of episodes
N_exec = 1; %20;
plotme = 0;
animateme=0;
animate_freq = 25;
rand('state',1000*sum(clock));
avgr_on = 1;

rews = zeros(N_exec,N_eps);   
traj = zeros(params.N_state_dim,N_eps_length);
act_trace = zeros(1,N_eps_length);

for ex=1:N_exec

    %create history storage
    %SART = zeros(N_eps * N_eps_length, params.N_state_dim+3);
    %actIndex = params.N_state_dim+1;
    %rewIndex = params.N_state_dim+2;
    %terIndex = params.N_state_dim+3;

    train_Reward = zeros(1,N_eps);
    train_Steps = zeros(1,N_eps);
     
 
    gpr2 = onlineGP.onlineGP_MA(params.N_act,params.rbf_mu,params.sigma,params.N_budget,params.tol,params.A);
    gpr_empty = onlineGP.onlineGP_MA(params.N_act,params.rbf_mu,params.sigma,params.N_budget,params.tol,params.A);
    %averager stuff
    avgr1 = onlineGP.onlineAVGR_MA(params.N_act,params.rbf_mu/4,params.sigma,params.N_budget,params.tol,params.Lip);
    timer = zeros(N_exec,N_eps * N_eps_length);
    
    if(updateStyle == -1)
        gpr1 = gpr2;
    end


    rew_exec_ind = [];
    %eval_counter = 0;

    allCount = 0;

    
    for j = 1:N_eps
        tic
        % Reset the initial state
        s_old = params.s_init;

        [qstart,action] = avgr1.getMax(s_old);
        
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
            
            startTime = cputime;

            %not using the variance here, because things will be set
            %optimistically already                
            [Q_opt,action] = avgr1.getMax(s_old);
            
            
            % Next State
            s_new = Generic_trans(s_old,action,params);
            
            %save trajectory on episode 100
            if mod(j,animate_freq) == 0
                traj(:,k) = s_old;
                act_trace(1,k) = action;
            end
            
            %s_indx_new = DiscretizeState(s_new',statelist);
            %Calculate The Reward
            [rew,breaker] = Generic_rew(s_new,action,params);
            rew = rew - RMAX;
            rew;
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
                y = y + (params.gamma * avgr1.getMax(s_new));
            end

            %subtract prior            
            temp = avgr1.predict(s_old,action);
            y = y-temp;
            gpr2.update(s_old,action,y);
            
            
            [act_val1,act_val_var1] =  avgr1.predict(s_old,action);
            
            %fprintf('state %d %d (%d) gp1: %f  %f gp2 %f %f \n',s_old(1), s_old(2), action_indx, act_val1, act_val_var1, act_val2, act_val_var2);
            firstCopy = 0;

                   


            %get values from the two GPs
            swap_done = 0;

            mu_IS1 =  avgr1.predict(s_old,action);        
            [mu_IS2,S_IS2] =  gpr2.predict(s_old,action);
            %do piecewise swap
            if ((mu_IS2 < -2*epsilon1) && (S_IS2 < SIGMA_TOL) ) 
                %need to add a BV at locations of GP2, doesn't
                %change function value though (or it shouldn't

                if avgr_on
                    avgr1.update(s_old,action,mu_IS1+mu_IS2);
                else
                    before = gpr1.predict(s_old,action);
                    gpr1.update(BV2(:,ii),action,before);
                end



                swap_done = 1;
                swap_counter = swap_counter + 1;
            end







            %update?
            if swap_done
             %old code 
             %for aa = 1:params.N_act;
             %  gpr2.copy(gpr_empty,aa);

             %end
             gpr2.reinitAll();
             
            end

        % if reachs the goal breaks the episode
            if breaker
                break;
            end

            s_old = s_new;
             endTime = cputime;
            timer(ex,allCount) = (endTime - startTime);


        end

        if(swap_counter > 0)
                fprintf('swapped %d times!\n',swap_counter);
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
        rews = zeros(N_exec,N_eps);   
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
              if avgr_on
                 [mean_post] = avgr1.getMax(x); 
              else
                 [mean_post] = gpr1.getMax(x);
              end
              %[mean_post] = gpr2.getMax(x); 
              mean_post = reshape(mean_post,21,21);

              surf(XX,YY,mean_post)
              pause(0.00000001);
              %Bellman error plot
              %{
              figure(2)
                [XX YY] = meshgrid(linspace(0,1,21));
              X = XX(:);
              Y = YY(:);
                %subplot(1,nmodels,iii)
              %gpr_temp = gpr1.getGP(iii);  
              x = [X';Y'];
              qt = gpr1.getMax(x);
              qt = reshape(qt,21,21);
              qt2 = zeros(21,21);
              iind = 1;
              for iii=0:.05:1
                  jind = 1;
                  for jjj =0:.05:1

                      qt2(iind,jind) = Generic_rew([iii;jjj],1,params) + params.gamma * gpr1.getMax(Generic_trans([iii;jjj],1,params));
                      for a = 2:params.N_act
                          [r,b] = Generic_rew([iii;jjj],a,params);
                          if(b)
                              qt2(iind,jind) = max([qt2(iind,jind),  r]);
                          else
                               qt2(iind,jind) = max([qt2(iind,jind),  r + params.gamma * gpr1.getMax(Generic_trans([iii;jjj],a,params))]);
                          end
                      end
                      jind = jind + 1;
                  end
                  iind = iind + 1;
              end


              surf(XX,YY,qt - qt2)
              pause(0.00000001);
              %}
        end

       if(breaker)
        fprintf('reached goal!!!!! in %d steps\n', step_counter);
       else
           fprintf('episode cap reached, chose action %d', action);
           %if(swap_counter == 0)
           %    s_old
           %end
        %fprintf('episode cap reached at [%d  %d] with values %f % f\n', s_old(1), s_old(2), gpr1.getMax(s_old), gpr2.getMax(s_old));
       end
        toc
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


filename = 'Lip_13';

save(filename,'mean_Reward', 'std_Dev_Reward', 'rews', 'timer');


