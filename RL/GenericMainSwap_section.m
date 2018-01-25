
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
%clear;
countSteps = 1;

%fh1 = figure;
%fh2 = figure;


%params.env = 'bandit';
%params.env = 'acrobot';
%params.env = 'grid';
params.env = 'puddle';
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

    N_eps_length =1; % Length of an episode

 elseif(strcmp(params.env,'acrobot')==1)

     %addpath('C:\cygwin\home\Tom\mit\gpq\GPQlearning\acrobot');
     epsilon1 = 0.01;
    if(updateStyle == 2)
        epsilon1 = 1; %threshold basis points needs to come down by before we do a check
    end

%GP params'



    gpbandwidth = 0.25;
    gpamptitude = 1.0; %NOTE: This parameter is controlling the initial variance estimate %1
    gpnoise = sqrt(0.1); 
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
    epsilon1 = 0.1;
    params.N_grid = 5;
    if(updateStyle == 2) %cautious
        epsilon1 = 0.5; %threshold basis points needs to come down by before we do a check
    end

    
    gpbandwidth = 0.15; %0.15;
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
 
 
 
 
 %averager stuff
 
 
 
 firstCopy =1;

%domain params
SNR = gpnoise;


%experiment parameters
N_eps = 400; % Number of episodes
N_exec = 1; %20;
plotme = 1;

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
    gpr2 = onlineGP.onlineGP_MA(params.N_act,params.rbf_mu,params.sigma,params.N_budget,params.tol,params.A);
    
    timer = zeros(N_exec,N_eps * N_eps_length);
    
    if(updateStyle == -1)
        gpr1 = gpr2;
    end


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
        s_indx_old = 1;%DiscretizeState(s_old',statelist);
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
            
            startTime = cputime;

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
                y = y + (params.gamma * gpr1.getMax(s_new));
            end

            gpr2.update(s_old,action_indx,y);

            %if(gpr2.predict(s_old,action_indx

            [act_val2,act_val_var2] =  gpr2.predict(s_old,action);
            [act_val1,act_val_var1] =  gpr1.predict(s_old,action);
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

                if(updateStyle == 1  && act_val2 + act_val_var2 < act_val1 - act_val_var1)
                    fprintf('significant change for action %d!  Making the swap!!!!!!!!!!!!!!!!!!!!\n', action_indx);
                    gpr1.copy(gpr2);      
                    gpr2.reinitCovar(params);

                elseif(updateStyle ==3 || updateStyle == 2) %KL divergence check on basis vectors      
                    SIG = params.rbf_mu;                
                    %SNR = 0.1; %params.sigma;
                    %collect the basis vectors

                    %cs1 = gpr1.get('current_size');
                    cs2 = gpr2.getField('current_size');

                    %get values from the two GPs
                    swap_done = zeros(1,params.N_act);
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

                            %new stuff, check at locations

                            [mu_IS2, S_IS2] =  gpr2.predict(s_old,jj);
                            [mu_IS1, S_IS1] =  gpr1.predict(s_old,jj);
                            

                            %do piecewise swap
                            
                            if ( ( -(mu_IS2-mu_IS1) > 2*epsilon1) && (S_IS2 < 1/8) ) %SNR^2/4))
                                %need to add a BV at locations of GP2, doesn't
                                %change function value though (or it shouldn't
                                before = gpr1.predict(s_old,jj);
                                gpr1.update(s_old,jj,before);

                                %TODO: NEED TO REPLACE ALL BVs AFTER CHECKS!

                                gpr1.replace_BV(s_old,mu_IS1-epsilon1,jj);
                                %gpr1.replace_BV(s_old,mu_IS2+epsilon1,jj);
                                %gpr1.replace_BV_optimist(BV2(:,ii),mu_IS2(ii)+epsilon1,epsilon1,jj);
                                swap_done(jj) = 1;
                                swap_counter = swap_counter + 1;
                                %this break will enforce only one swap per step
                                %break;


                                %gpr1.copy(gpr2);
                                %gpr2.reinitCovar();
                            end
                            



                        end

                        %update?
                        if sum(swap_done) > 0  %if anybody swapped we need to reset all actions
                            for jj = 1:params.N_act
                               %copies gp1 into gp2
                                %{
                                gpr2 = onlineGP.onlineGP_MA(params.N_act,params.rbf_mu,params.sigma,params.N_budget,params.tol,params.A);
                                for jj=1:params.N_act
                                    BV = [gpr1.getOnlyAct('BV',jj)]; 
                                    cs1 = gpr1.getOnlyAct('current_size',jj);
                                    for b=1:cs1
                                        y = gpr1.predict(BV(:,b),jj);
                                        gpr2.update(BV(:,b),jj,y(1));
                                    end
                                end
                               %}
                                gpr2.copy(gpr1,jj);
                                gpr2.reinitCovar(); %everybody opens up, not just the one who swapped
                            end

                        end
                        
                        
                        
                        %old stuff, checks at BVs
                        %{
                        BV2 = [gpr2.getOnlyAct('BV',jj)]; 
                        BV2 = BV2(:,1:csa2);



                        %Checked: took care of non-full BV and correct indexing
                        %of actions


                        [mu_IS2, S_IS2] =  gpr2.predict(BV2,jj);
                        [mu_IS1, S_IS1] =  gpr1.predict(BV2,jj);

                      if(min(mu_IS2) < -100 || min(mu_IS1) < -100)
                           %fprintf('went below vmin\n');

                       end

                        if(~ (size(BV2',1) == size(unique(BV2','rows'),1)))
                                  fprintf('Repeated basis vector!!!');
                                    BV2
                                    jj
                        end

                        %do piecewise swap
                        for ii = 1:length(mu_IS2)
                            if ( ( abs(mu_IS2(ii)-mu_IS1(ii)) > 2*epsilon1) && (S_IS2(ii,ii) < 1/2) ) %SNR^2/4))
                                %need to add a BV at locations of GP2, doesn't
                                %change function value though (or it shouldn't
                                before = gpr1.predict(BV2(:,ii),jj);
                                gpr1.update(BV2(:,ii),jj,before);

                                %TODO: NEED TO REPLACE ALL BVs AFTER CHECKS!


                                gpr1.replace_BV(BV2(:,ii),mu_IS2(ii)+epsilon1,jj);
                                %gpr1.replace_BV_optimist(BV2(:,ii),mu_IS2(ii)+epsilon1,epsilon1,jj);
                                swap_done(jj) = 1;
                                swap_counter = swap_counter + 1;
                                %this break will enforce only one swap per step
                                %break;


                                %gpr1.copy(gpr2);
                                %gpr2.reinitCovar();
                            end
                        end



                    end

                    %update?
                    if sum(swap_done) > 0  %if anybody swapped we need to reset all actions
                        for jj = 1:params.N_act
                           %copies gp1 into gp2
                            %{
                            gpr2 = onlineGP.onlineGP_MA(params.N_act,params.rbf_mu,params.sigma,params.N_budget,params.tol,params.A);
                            for jj=1:params.N_act
                                BV = [gpr1.getOnlyAct('BV',jj)]; 
                                cs1 = gpr1.getOnlyAct('current_size',jj);
                                for b=1:cs1
                                    y = gpr1.predict(BV(:,b),jj);
                                    gpr2.update(BV(:,b),jj,y(1));
                                end
                            end
                           %}
                            gpr2.copy(gpr1,jj);
                            gpr2.reinitCovar(); %everybody opens up, not just the one who swapped
                        end

                    end
                        %}


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

        %plot
        if plotme && mod(j,10) == 0
                    %plot
            figure(1);
            [XX YY] = meshgrid(linspace(0,1,21));
              X = XX(:);
              Y = YY(:);
                %subplot(1,nmodels,iii)
              %gpr_temp = gpr1.getGP(iii);  
              x = [X';Y'];
              [mean_post] = gpr1.getMax(x);
              %[mean_post] = gpr1.getGP(4).predict(x);
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

        %alpha = gpr.get('alpha_store');
        %if j>1
        %    diff_alpha_norm(j-1) = norm(alpha-alpha_old,'fro');
        %end
        %alpha_old = alpha;
       if(breaker)
        fprintf('reached goal!!!!! in %d steps\n', step_counter);
       else
           fprintf('episode cap reached, chose action %d', action);
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



save(filename,'mean_Reward', 'std_Dev_Reward', 'rews', 'timer');
