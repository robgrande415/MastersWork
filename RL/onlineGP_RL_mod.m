%============================== onlineGP ==================================
%
%  This code implements the sparse online GP algorithm presented in the
%  reference for basic GP regression with a Gaussian kernel.
%
%  This code is currently designed strictly for Gaussin kernels: if
%  you wish to extend it for non-Gaussian kernels, you MUST change
%  the values of k* away from unity!
%
%  Reference(s):
%    Sparse Online Gaussian Processes -Csato and Opper, Tech Report
%    Csato's thesis
%
%  Inputs:
%    sigma  	 - bandwidth for the Gaussian kernel; either
%                  1 x 1 scalar or
%                  1 x d vector
%    noise      -  parameter for noise in GP; assumed given by oracle
%    m          -  the size of your budget
%    tol        -  tolerance for projection residual
%
%  Outputs:
%    see functions
%
%============================== onlineGP ==================================
%
%  Name:		onlineGP.m
%
%  Authors: 		Hassan A. Kingravi, Girish Chowdhary
%
%  Created:  	2011/02/27
%  Modified: 	2012/02/29
%
%============================== onlineGP ==================================
function oGP = onlineGP_RL_mod(sigma,noise,max_points,tol,params)


if params.state_action_slicing_on==0
    
    BV           = [];            % Basis vector set
    K            = [];            % Kernel matrix
    alpha        = [];            % mean parameter
    C            = [];            % inverted covariance matrix
    Q            = [];            % inverted Gram matrix
    current_size = [];
    obs          = [];
    rew          = [];
    index        = 1;
elseif params.state_action_slicing_on==1
   BV_store           = zeros(params.N_state_dim,max_points*params.N_act);            % Basis vector set
   K_store            = zeros(max_points,max_points*params.N_act);            % Kernel matrix
   alpha_store        = zeros(params.N_act,max_points);            % mean parameter
   C_store            = K_store;            % inverted covariance matrix
   Q_store            = K_store;            % inverted Gram matrix
   current_size_store = zeros(1,params.N_act);% min size is 1
   obs_store          = zeros(params.N_act,max_points);
   rew_store          = zeros(params.N_act,max_points);
   index_store        = ones(1,params.N_act);
    
end


%==[2] Setup the interface functions.
%
oGP.predict = @predict;
oGP.update  = @update;
oGP.get = @oGP_get;
oGP.predict_var_improv=@predict_var_improv;
oGP.kernel = @kernel;
oGP.copy  = @copy;
oGP.reinitCovar = @reinitCovar;
oGP.getOnlyAct = @oGP_getOnlyAct;
oGP.replace_BV = @replace_BV;

function reinitCovar(params)
    %if(isfield(params, 'old_current_size') && sum(params.old_current_size) ~= 0)
        
        C_store = zeros(max_points,max_points*params.N_act);   %current size (assuming this is the size of C)   
        
        %reinitiallize for every action (won't work on the full matrix
         %K_store            = zeros(max_points,max_points*params.N_act); 
        for a=1:params.N_act
         %   bot = max_points * (a-1) + 1; 
         if params.state_action_slicing_on==1    
            [BV,K,alpha,C,Q,current_size,obs,rew,index]=assign_vars(a);
         end   
         %current_size=current_size_store(a);
            %starter = max_points*(a-1)+1;
            %ender = max_points*(a-1)+current_size;
            %BV=BV_store(:,starter:ender);
            
            
            K(1:current_size,1:current_size) = kernel(BV,BV,sigma,params);
            Q(1:current_size,1:current_size) = K\eye(size(K,1)); %inv(K); %changed on Hassan's visit 8/1/13
            
            
            if params.state_action_slicing_on==1
                [BV_store,K_store,alpha_store,C_store,Q_store,current_size_store,obs_store,rew_store,index_store]=assign_vars_back(a,BV_store,K_store,alpha_store,C_store,Q_store,current_size_store,obs_store,rew_store,index_store);
            end
            %K_store = kernel(BV,BV,sigma,params); 
         %  top = bot + max_points -1;
            %TODO: make this much faster
        %for i=1:size(BV_store,2)
        %    for j=1:size(BV_store,2)
         %      K_store(i,j) = kernel(BV_store(:,i),BV_store(:,j),sigma,params);  %BV, BV, sigma
         %   end
        end
        
        %K = K + noise^2*eye(current_size);
        
        
    %end
end

function copy(oldGP,newMu)
        BV_store           = oldGP.get('BV_store');
        K_store            = oldGP.get('K_store'); 
        alpha_store        = oldGP.get('alpha_store')+0.01; 
        C_store            = oldGP.get('C_store'); 
        Q_store            = oldGP.get('Q_store');
        current_size_store = oldGP.get('current_size'); 
        obs_store          = oldGP.get('obs_store'); 
        rew_store          = oldGP.get('rew_store');
        index_store        = oldGP.get('index_store');
        
        %RCG
        %modifies alphas
        if nargin > 1
            cs1 = oldGP.get('current_size');
            if sum(cs1==0) ==0
               for ind = 1:size(alpha_store,1)
                   csa1 = cs1(ind);
                   BV1 = oldGP.getOnlyAct('BV_store',ind);
                   BV1 = BV1(:,1:csa1);
                   S1 = kernel(BV1,BV1,sigma,params);
                  
                   alpha_store(ind,1:csa1) = (S1\newMu{ind})';

               end
            end
        end
end

%------------------------------- predict -------------------------------
%
%  Given a new datapoint, predict a new value based on current model
%
%(
    function [f,var_x] = predict(x,params)
        
        if params.state_action_slicing_on==0
            k = kernel(x,BV,sigma,params)';
            f = k'*alpha;
            var_x = kernel(x,x,sigma,params) + k'*C*k;
        elseif params.state_action_slicing_on==1
            a=getaction(x,params);  %assumes the same action for all
            x=x(1:params.N_state_dim,:);
            
            [BV,K,alpha,C,Q,current_size,obs,rew,index]=assign_vars(a);
            if(current_size == 0)
                f = zeros(size(x,2),1); 
                var_x = noise + params.A;  
            else
                k = kernel(x,BV,sigma,params)';
                %             kt=getactivebasis(k,params);
                %             alphat=getactiveweights(alpha,params);
                f = k'*alpha;
                kxx =  kernel(x,x,sigma,params); % * params.A;
                %             kxx=getactivebasis(kxx,a,params);
                var_x=kxx + k'*C(1:current_size,1:current_size)*k;%needs to be sent out for each action
                %  [BV_store,K_store,alpha_store,C_store,Q_store,current_size_store,obs_store,rew_store,index_store]=assign_vars_back(a,BV_store,K_store,alpha_store,C_store,Q_store,current_size_store,obs_store,rew_store,index_store);
            end
         end
    end

%------------------------------- predict_var_improv-------------------------------
%
%  Given a new datapoint, predicts the improvement in variance if that
%  measurement were taken
%
%(
    function [var_improv_x] = predict_var_improv(x)
        
        if params.state_action_slicing_on==0
            k = kernel(x,BV,sigma,params)';
            var_improv_x = k'*C*k;
        elseif params.state_action_slicing_on==1
            a=getaction(x,params);
            x=x(1:params.N_state_dim);
            [BV,K,alpha,C,Q,current_size,obs,rew,index]=assign_vars(a);
            
            k = kernel(x,BV,sigma,params)';
            kxx = kernel(x,x,sigma,params);
            %             kxx=getactivebasis(kxx,a,params);
            var_improv_x=k'*C(1:current_size,1:current_size)*k;%needs to be sent out for each action
           
            %  [BV_store,K_store,alpha_store,C_store,Q_store,current_size_store,obs_store,rew_store,index_store]=assign_vars_back(a,BV_store,K_store,alpha_store,C_store,Q_store,current_size_store,obs_store,rew_store,index_store);
        end
    end


%------------------------------- update --------------------------------
%
%  Given a new data pair, update the model; remember, this is passed
%  columnwise
%(
    function update(x,reward,Qmax,params)
        % first compute simple upate quantities
        a=getaction(x,params);
        if params.state_action_slicing_on==1
            x=x(1:params.N_state_dim);
            [BV,K,alpha,C,Q,current_size,obs,rew,index]=assign_vars(a);
        end
        
        y = reward+params.gamma*Qmax;
        
        if(current_size == 0)
            
            
            
            
                BV(:,index) = x;
                rew(index)=reward;
                obs(index)=y;
                current_size = 1;
                I = eye(current_size);
                K = kernel(x,x,sigma,params);
                Q = K\I; % inverted gram matrix
                %K = K + noise*eye(current_size);
                C = (K+noise*eye(current_size))\I; % less numerically stable than cholesky
                alpha = C*y';
           
                
                %if params.state_action_slicing_on==1
                 %   [BV_store,K_store,alpha_store,C_store,Q_store,current_size_store,obs_store,rew_store,index_store]=assign_vars_back(a,BV_store,K_store,alpha_store,C_store,Q_store,current_size_store,obs_store,rew_store,index_store);
                %end
                %return;
        end
            k_t1 = kernel(x,BV,sigma,params)';   % pg 9, eqs 30-31
            %         if params.state_action_slicing_on==1
            %          k_t1=getactivebasis(k_t1,a,params);%get the active basiss for the current action if state space slicing is on
            %          alpha=getactiveweights(alpha,a,params);%get the active weights corresponding to active basis
            %         end
            noise_x = noise + k_t1'*C*k_t1+ params.A;
            q_t1 = (y - k_t1'*alpha)/(noise_x + noise);
            r_t1 = -1/(noise_x + noise);

            % compute residual projection update quantities
            e_t1 = Q*k_t1; %residual vector pg 6, eq 16
            %EDIT: 7/19 replaced params.A with 1
            gamma_t1 = double(params.A-k_t1'*e_t1); %novelty of new point w.r.t RKHS: pg 7, eq 23
            eta_t1 = 1/(1+gamma_t1*r_t1); %numerical stability parameter

         
        
        
        if gamma_t1 < tol
            % in this case, addition of point to basis doesn't help much, so
            % don't add it, and compute update quantities in terms of old vectors
            % note that data, obs and gram matrix inverse not updated
            s_t1 = C*k_t1 + e_t1;                  %pg 5, eqs 9, but modified
            alpha = alpha + q_t1*eta_t1*s_t1;
            C = C + r_t1*eta_t1*(s_t1*s_t1');
            if params.state_action_slicing_on==1 %assign quantities and leave
                [BV_store,K_store,alpha_store,C_store,Q_store,current_size_store,obs_store,rew_store,index_store]=assign_vars_back(a,BV_store,K_store,alpha_store,C_store,Q_store,current_size_store,obs_store,rew_store,index_store);
            end
            %  obs=recalculate_Qstar(BV,rew,obs,current_size,params); %this
            %  doesn't do anything because the way alpha is updated. Its
            %  updated whether or not the point gets stored
            
        else
            % in this case, you need to add the points
            current_size = current_size + 1;
            
            
            %in this case, you can simply add the points
            s_t1 = [C*k_t1; 1];
            alpha = [alpha; 0] + q_t1*s_t1;
            C = [C zeros(current_size-1,1); zeros(1,current_size)] + r_t1*(s_t1*s_t1');
            
            % update basis vectors and observations
            BV = [BV x];
            if(~ (size(BV',1) == size(unique(BV','rows'),1)))
                fprintf('Repeated basis vector!!!');
                BV
                a
                
            end
            
            obs = [obs, y];
            rew = [rew, reward];
            
            % update Gram matrix and inverse
            K = [K k_t1; k_t1' params.A];  
%             [L,U] = lu(K);
%             Y = L\eye(size(K,1));
%             Q = U\Y;
            Q = K\eye(size(K,1));
            if current_size <= max_points
                %do nothing
            else
                if params.sparsification==1
                    % now you must delete one of the basis vectors; follow figure 3.3
                    % first, compute which vector is least informative (1), pg 8, eq 27
                    scores = zeros(1,current_size);
                    for i=1:current_size
                        scores(i) = abs(alpha(i))/Q(i,i);
                    end
                    
                    %find index of minimum vector
                    [val index] = min(scores);
                    
                elseif params.sparsification==2
                    %just implement a windowed buffer
                    index=index+1;
                    if index==max_points
                        index=1;
                    end
                end
                
                %now begin update given in (1), pg 8, eq 25
                
                %first compute scalar parameters
                a_s = alpha(index);
                c_s = C(index,index);
                q_s = Q(index,index);
                
                %compute vector parameters
                C_s = C(:,index);
                C_s(index) = [];
                Q_s = Q(:,index);
                Q_s(index) = [];
                
                %shrink matrices
                alpha(index) = [];
                C(:,index)   = [];
                C(index,:)   = [];
                Q(:,index)   = [];
                Q(index,:)   = [];
                K(:,index)   = [];
                K(index,:)   = [];
                
                %finally, compute updates
                alpha = alpha - (a_s/q_s)*(Q_s);
                C = C + (c_s/(q_s^2))*(Q_s*Q_s') - (1/q_s)*(Q_s*C_s' + C_s*Q_s');
                Q = Q - (1/q_s)*(Q_s*Q_s');
                
                current_size = current_size - 1;
                BV(:,index) = [];
                obs(index) = [];
                rew(index) = [];
            end
            
            %       c = (1/gamma_t1);
            %       mc = -c;
            %       Q = [(Q + c*(e_t1*e_t1')) mc*e_t1; mc*e_t1' c];          e
            %recalculate the estimate of Qstar
            if params.state_action_slicing_on==1
                [BV_store,K_store,alpha_store,C_store,Q_store,current_size_store,obs_store,rew_store,index_store]=assign_vars_back(a,BV_store,K_store,alpha_store,C_store,Q_store,current_size_store,obs_store,rew_store,index_store);
            end
            %              obs=recalculate_Qstar(BV,rew,obs,current_size,params);
        end
        
    end

%------------------------------- replace_BV --------------------------------
%
%  Given a new data pair, update the model; remember, this is passed
%  columnwise
%(
    function replace_BV(x,y,params)
        % first compute simple upate quantities
        a=getaction(x,params);
        if params.state_action_slicing_on==1
            x=x(1:params.N_state_dim);
            [BV,K,alpha,C,Q,current_size,obs,rew,index]=assign_vars(a);
        end
        
        %y = reward+params.gamma*Qmax;
        
       %delete stuff if it has BVs
       if isempty(BV) == 0
          
        if params.sparsification==1
            % now you must delete one of the basis vectors; follow figure 3.3
            %find closest BV
            %NOTE: should be weighted distance
            x
            BV
            dist_BV_x = sum( (repmat(x,1,size(BV,2)) - BV).^2,1);
            [~, index] = min(dist_BV_x);
        elseif params.sparsification==2
            %just implement a windowed buffer
            index=index+1;
            if index==max_points
                index=1;
            end
        end

        %now begin update given in (1), pg 8, eq 25

        %first compute scalar parameters
        a_s = alpha(index);
        c_s = C(index,index);
        q_s = Q(index,index);

        %compute vector parameters
        C_s = C(:,index);
        C_s(index) = [];
        Q_s = Q(:,index);
        Q_s(index) = [];

        %shrink matrices
        alpha(index) = [];
        C(:,index)   = [];
        C(index,:)   = [];
        Q(:,index)   = [];
        Q(index,:)   = [];
        K(:,index)   = [];
        K(index,:)   = [];

        %finally, compute updates
        alpha = alpha - (a_s/q_s)*(Q_s);
        C = C + (c_s/(q_s^2))*(Q_s*Q_s') - (1/q_s)*(Q_s*C_s' + C_s*Q_s');
        Q = Q - (1/q_s)*(Q_s*Q_s');

        current_size = current_size - 1;
        BV(:,index) = [];
        obs(index) = [];
        rew(index) = [];
                
                
                
                
        
            k_t1 = kernel(x,BV,sigma,params)';   % pg 9, eqs 30-31
            %         if params.state_action_slicing_on==1
            %          k_t1=getactivebasis(k_t1,a,params);%get the active basiss for the current action if state space slicing is on
            %          alpha=getactiveweights(alpha,a,params);%get the active weights corresponding to active basis
            %         end
            noise_x = noise + k_t1'*C*k_t1+ params.A;
            q_t1 = (y - k_t1'*alpha)/(noise_x + noise);
            r_t1 = -1/(noise_x + noise);

            % compute residual projection update quantities
            e_t1 = Q*k_t1; %residual vector pg 6, eq 16
            %EDIT: 7/19 replaced params.A with 1
            gamma_t1 = double(params.A-k_t1'*e_t1); %novelty of new point w.r.t RKHS: pg 7, eq 23
            eta_t1 = 1/(1+gamma_t1*r_t1); %numerical stability parameter
        end
            
            k_t1 = kernel(x,BV,sigma,params)'; 
            % in this case, you need to add the points
            current_size = current_size + 1;
            
            
            %in this case, you can simply add the points
            s_t1 = [C*k_t1; 1];
            alpha = [alpha; 0] + q_t1*s_t1;
            C = [C zeros(current_size-1,1); zeros(1,current_size)] + r_t1*(s_t1*s_t1');
            
            % update basis vectors and observations
            BV = [BV x];
            if(~ (size(BV',1) == size(unique(BV','rows'),1)))
                fprintf('Repeated basis vector!!!');
                BV
                a
                
            end
            
            obs = [obs, y];
            reward = y;
            rew = [rew, reward];
            
            % update Gram matrix and inverse
            K = [K k_t1; k_t1' params.A];  
%             [L,U] = lu(K);
%             Y = L\eye(size(K,1));
%             Q = U\Y;
            Q = K\eye(size(K,1));
      
            if params.state_action_slicing_on==1
                [BV_store,K_store,alpha_store,C_store,Q_store,current_size_store,obs_store,rew_store,index_store]=assign_vars_back(a,BV_store,K_store,alpha_store,C_store,Q_store,current_size_store,obs_store,rew_store,index_store);
            end
         
        
        
        
        
    end

      
      
        
          


%)
%-------------------------------- get --------------------------------
%
%  Get a requested member variable, choose the columns by act
%
%(

    function mval = oGP_getOnlyAct(mfield, act)
        mval1 = oGP_get(mfield);
        mval = mval1(:,((act-1)*params.N_budget+1):act*params.N_budget);
    end

%)
%-------------------------------- get --------------------------------
%
%  Get a requested member variable.
%
%(

    function mval = oGP_get(mfield)
        
        switch(mfield)
            case {'basis','BV'}
                mval = BV;
            case {'obs'}
                mval = obs;
            case {'K','kernel'}
                mval = K;
            case {'Q'}
                mval = Q;
            case {'current_size','size','current size'}
                mval = current_size_store;
            case {'max_points'}
                mval = max_points;
            case {'alpha'}
                mval=alpha;
            case {'BV_store'}
                mval=BV_store;
            case {'K_store'}
                mval=K_store;
            case {'alpha_store'}
                mval=alpha_store;
            case {'Q_store'}
                mval=Q_store;
            case {'obs_store'}
                mval=obs_store;
            case {'rew_store'}
                mval=rew_store;
            case {'index_store'}
                mval=index_store;
            case {'C_store'}
                mval = C_store;
        end
        
    end
%)
    function obs=recalculate_Qstar(inputs,rew,obs,current_size,params);
        %           if params.state_action_slicing_on==1
        %             x=x(1:params.N_state_dim);
        %             [BV,K,alpha,C,Q,current_size,obs,rew,index]=assign_vars(a);
        %         end
        for ii=1:current_size
            N_act = params.N_act;
            
            act_val = zeros(1,N_act);
            for l=1:N_act
                
                a = l;
                x=[inputs(1:params.N_state_dim,ii);a];
                [mean_post, var_post] = predict(x);
                act_val(l) = mean_post;
                
            end
            
            [Q_opt,nothing] = max(act_val);
            
            obs(ii)=rew(ii)+params.gamma*Q_opt;
        end
    end
    function [BV,K,alpha,C,Q,current_size,obs,rew,index]=assign_vars(a)
        %assign variables according to the appropriate GP based on the
        %action
        %         BV=BV_store(:,max_points*(a-1)+1:max_points*a);
        %         K=K_store(:,max_points*(a-1)+1:max_points*a);
        %         alpha=alpha_store(a,:)';%make it a column vector
        %         C=C_store(:,max_points*(a-1)+1:max_points*a);
        %         Q=Q_store(:,max_points*(a-1)+1:max_points*a);
        %         current_size=current_size_store(a);
        %         obs=obs_store(a,:);
        %         rew=rew_store(a,:);
        %         index=index_store(a);

    current_size=current_size_store(a);

BV=BV_store(:,max_points*(a-1)+1:max_points*(a-1)+current_size);
        K=K_store(1:current_size,max_points*(a-1)+1:max_points*(a-1)+current_size);
        alpha=alpha_store(a,1:current_size)';%make it a column vector
        C=C_store(1:current_size,max_points*(a-1)+1:max_points*(a-1)+current_size);
        Q=Q_store(1:current_size,max_points*(a-1)+1:max_points*(a-1)+current_size);
        
        obs=obs_store(a,1:current_size);
        rew=rew_store(a,1:current_size);
        index=index_store(a);
    end
    function [BV_store,K_store,alpha_store,C_store,Q_store,current_size_store,obs_store,rew_store,index_store]=assign_vars_back(a,BV_store,K_store,alpha_store,C_store,Q_store,current_size_store,obs_store,rew_store,index_store);
        %assign variables back to original storage vars according to the
        %action
        %         BV_store(:,max_points*(a-1)+1:max_points*a)=BV;
        %         K_store(:,max_points*(a-1)+1:max_points*a)=K;
        %         alpha_store(a,:)=alpha';%make it back a row vector
        %         C_store(:,max_points*(a-1)+1:max_points*a)=C;
        %         Q_store(:,max_points*(a-1)+1:max_points*a)=Q;
        %         current_size_store(a)=current_size;
        %         obs_store(a,:)=obs;
        %         rew_store(a,:)=rew;
        %         index_store(a)=index;
        
        current_size_store(a)=current_size;
        BV_store(:,max_points*(a-1)+1:max_points*(a-1)+current_size)=BV;
        K_store(1:current_size,max_points*(a-1)+1:max_points*(a-1)+current_size)=K;
        alpha_store(a,1:current_size)=alpha;%make it a column vector
        C_store(1:current_size,max_points*(a-1)+1:max_points*(a-1)+current_size)=C;
        Q_store(1:current_size,max_points*(a-1)+1:max_points*(a-1)+current_size)=Q;
        obs_store(a,1:current_size)=obs;
        rew_store(a,1:current_size)=rew;
        index_store(a)=index;
    end
    function reset_vars(params)
        BV_store           = zeros(params.N_state_dim,max_points*params.N_act);            % Basis vector set
        K_store            = zeros(max_points,max_points*params.N_act);            % Kernel matrix
        alpha_store        = zeros(params.N_act,max_points);            % mean parameter
        C_store            = K_store;            % inverted covariance matrix
        Q_store            = K_store;            % inverted Gram matrix
        current_size_store = ones(1,params.N_act);% min size is 1
        obs_store          = zeros(params.N_act,max_points);
        rew_store          = zeros(params.N_act,max_points);
        index_store        = ones(1,params.N_act);
    end
end

%============================ Helper Functions ===========================

%------------------------------- kernel ------------------------------
%y is a matrix of centers
%
%
function v =  kernel(xin,yin,sigma,params)

%  v = x'*y ;
if params.state_action_slicing_on == 0
    x=xin;
    y=yin;
    if(length(sigma) == 1) %same sigma
        d=x'*y;
        dx = sum(x.^2,1);
        dy = sum(y.^2,1);
        val = repmat(dx',1,length(dy)) + repmat(dy,length(dx),1) - 2*d;
        v = params.A*exp(-val./(2*sigma^2));
    else
        isigma = inv(diag(sigma.^2));
        d =  (x'*isigma)*y;
        dx = sum((x'*isigma)'.*x,1);
        dy = sum((y'*isigma)'.*y,1);
        val = repmat(dx',1,length(dy)) + repmat(dy,length(dx),1) - 2*d;
        v = params.A*exp(-val./2);
    end
elseif params.state_action_slicing_on==1
    x=xin(1:params.N_state_dim,:);
    %     a=getaction(xin,params);
    y=yin(1:params.N_state_dim,:);
    if(length(sigma) == 1) %same sigma
        d=x'*y;
        dx = sum(x.^2,1);
        dy = sum(y.^2,1);
        val = repmat(dx',1,length(dy)) + repmat(dy,length(dx),1) - 2*d;
        vtemp = params.A*exp(-val./(2*sigma^2));
    else
        %isigma = 1\diag(sigma.^2);
        isigma = inv(diag(sigma.^2));
        d =  (x'*isigma)*y;
        dx = sum((x'*isigma)'.*x,1);
        dy = sum((y'*isigma)'.*y,1);
        val = repmat(dx',1,length(dy)) + repmat(dy,length(dx),1) - 2*d;
        vtemp = params.A*exp(-val./2);
    end
    v=vtemp;
    %v = zeros(params.N_phi_s*params.N_act,1);
    %v(((a-1)*params.N_phi_s + 1):a*params.N_phi_s) = vtemp;
end
end

%------------------------------- getaction ------------------------------
%
%
%

function a= getaction(xin,params)
a=xin(params.N_state_dim+1:end,1);
end

%------------------------------- getactivebasis ------------------------------
%y is a matrix of centers
%
%
% function kout=getactivebasis(kin,a,params)
%     kout=kin(((a-1)*params.N_phi_s + 1):a*params.N_phi_s);
% end
%
% %------------------------------- getactiveweights ------------------------------
% %overloads getactivebasis
% %
% %
% function alphaout=getactiveweights(alphain,a,params)
%     alphaout=alphain(((a-1)*params.N_phi_s + 1):a*params.N_phi_s);
% end
% %------------------------------- getactiveweights ------------------------------
% %
% %
% %


%================================== kpca =================================
