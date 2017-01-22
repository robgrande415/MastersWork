function out = run_DPGP_MCMC(x,y,params,predict_on)

%main_DPGP 
ind = length(x);
alpha = params.kl_tol;

%init changepoints, uses one at beginning
num_chpt = 10;
chpt_sigma = 50;

num_clusters = num_chpt;
chpt_ind = randperm(ind-1);
chpt_ind = [1,sort(chpt_ind(1:num_chpt-1))];
chpt_cluster_id = 1:num_chpt;
cluster_pop = zeros(1,num_clusters);
cluster_id = zeros(1,ind);
%initialize as different functions
for i = 1:ind
    temp_ind = max(find(i >= chpt_ind));
     cluster_id(i) = chpt_cluster_id(temp_ind);
    cluster_pop(cluster_id(i)) = cluster_pop(cluster_id(i))+1;
end

first_pt = zeros(1,num_clusters);
%GPs
for i = 1:num_clusters
   gp_temp{chpt_cluster_id(i)} = onlineGP.onlineGP_PE(params.bandwidth,params.noise,params.max_points,params.tol);  
   cluster_pop(i) = sum(cluster_id == i);
end
for i = 1:ind
    if first_pt(cluster_id(i)) == 0
        gp_temp{cluster_id(i)}.process(x(:,i),y(i));
        first_pt(cluster_id(i)) = 1;
    else
        gp_temp{cluster_id(i)}.update(x(:,i),y(i));
    end
end

%MCMC
gp_llh = onlineGP.gp_regression(params.bandwidth,params.noise);
llh_last = -inf*ones(1,100);
counter = 0;
MAX_ITER=10000;
%loop and sample
for k = 1:MAX_ITER
    
    %for now just do shifting
    type_change = [];
    temp = rand;
    if temp >= 1/3
        type_change = 1;
    else
        type_change = 0;
    end
    
    %shift chpt
    if type_change ==1
        %sample new change point
        temp_i = randi(num_chpt-1)+1;
        chpt_ind_temp = chpt_ind;
        cluster_id_temp = cluster_id;
        chpt_cluster_id_temp = chpt_cluster_id;
        shift = randn*chpt_sigma;
        if temp_i == num_chpt
            if shift >0
                chpt_ind_temp(temp_i) = min([ind-1,chpt_ind(temp_i)+round(shift)]);
            else
                chpt_ind_temp(temp_i) = max([chpt_ind(temp_i-1)+2,chpt_ind(temp_i)+round(shift)]);
            end
        else
            if shift < 0
                chpt_ind_temp(temp_i) = max([chpt_ind(temp_i-1)+2,chpt_ind(temp_i)+round(shift)]);
            else
                chpt_ind_temp(temp_i) = min([chpt_ind(temp_i+1)-2,chpt_ind(temp_i)+round(shift)]);
            end
        end
        
        %get llh of old setup
        llh_old = 0;
        for i = 1:num_chpt-1
            x0 = x(:,chpt_ind(i):chpt_ind(i+1));
            y0 = y(chpt_ind(i):chpt_ind(i+1));
            if isempty(x0) == 0
                [mu, var] = gp_temp{chpt_cluster_id(i)}.predict(x0,'full');
                llh_old = llh_old + gp_llh.model_prob_covar(var, y0-mu');
            end
        end
        %get llh of new setup
        llh_new = 0;
        for i = 1:num_chpt-1
            x0 = x(:,chpt_ind_temp(i):chpt_ind_temp(i+1));
            y0 = y(chpt_ind_temp(i):chpt_ind_temp(i+1));
            if isempty(x0) == 0
                [mu, var] = gp_temp{chpt_cluster_id(i)}.predict(x0,'full');
                llh_new = llh_new + gp_llh.model_prob_covar(var, y0-mu');
            end
        end

        %get prior probabilities
        if temp_i == num_chpt
            psi_shift = (normcdf(chpt_ind(temp_i),ind,chpt_sigma)-normcdf(chpt_ind(temp_i),chpt_ind(temp_i-1),chpt_sigma))/...
                            (normcdf(chpt_ind_temp(temp_i),ind,chpt_sigma)-normcdf(chpt_ind_temp(temp_i),chpt_ind_temp(temp_i-1),chpt_sigma));
        else
            psi_shift = (normcdf(chpt_ind(temp_i),chpt_ind(temp_i+1),chpt_sigma)-normcdf(chpt_ind(temp_i),chpt_ind(temp_i-1),chpt_sigma))/...
                            (normcdf(chpt_ind_temp(temp_i),chpt_ind_temp(temp_i+1),chpt_sigma)-normcdf(chpt_ind_temp(temp_i),chpt_ind_temp(temp_i-1),chpt_sigma));
        end
    elseif type_change == 0
        %reassign to another cluster
        chpt_ind_temp = chpt_ind;
        chpt_cluster_id_temp = chpt_cluster_id;
        cluster_id_temp = cluster_id;
        temp_i = randi(num_chpt); %changepoint num
        chpt_cluster_id_temp(temp_i) = randi(num_clusters); %reassign ; +1 allows more clusters
        %for i = 1:ind
        %    temp_ind = max(find(i >= chpt_ind));
        %    cluster_id(i) = chpt_cluster_id_temp(temp_ind);
        %    cluster_pop(cluster_id(i)) = cluster_pop(cluster_id(i))+1;
        %end
        
        %get llh of old setup
        llh_old = 0;
        for i = 1:num_clusters
            x0 = x(:,cluster_id==i);
            y0 = y(cluster_id==i);
             if isempty(x0) == 0
                [mu, var] = gp_temp{chpt_cluster_id(i)}.predict(x0,'full');
                llh_old = llh_old + gp_llh.model_prob_covar(var, y0-mu');
             end
        end
        
        
        num_clusters = max(chpt_cluster_id_temp);
        first_pt = zeros(1,num_clusters);
        %GPs
        for i = 1:ind
            temp_ind = max(find(i >= chpt_ind));
            cluster_id_temp(i) = chpt_cluster_id_temp(temp_ind);
        end
        for i = 1:num_clusters
           gp_temp_new{i} = onlineGP.onlineGP_PE(params.bandwidth,params.noise,params.max_points,params.tol);  
        end
        for i = 1:ind
            if first_pt(cluster_id_temp(i)) == 0
                gp_temp_new{cluster_id_temp(i)}.process(x(:,i),y(i));
                first_pt(cluster_id_temp(i)) = 1;
            else
                gp_temp_new{cluster_id_temp(i)}.update(x(:,i),y(i));
            end
        end
        %get llh of new setup
        llh_new = 0;
        for i = 1:num_clusters
            x0 = x(:,cluster_id_temp==i);
            y0 = y(cluster_id_temp==i);
             if isempty(x0) == 0
                [mu, var] = gp_temp_new{chpt_cluster_id_temp(i)}.predict(x0,'full');
                llh_new = llh_new + gp_llh.model_prob_covar(var, y0-mu');
             end
        end
        
        %not quite correct, but hopefully close enough
        psi_shift = (sum(cluster_id == chpt_cluster_id_temp(temp_i)))/sum(cluster_id == chpt_cluster_id(temp_i));
        
        if psi_shift ==0
            uni_clusters_old= unique(chpt_cluster_id);
            uni_clusters_new = unique(chpt_cluster_id_temp);
            n_uni_clusters_old = length(uni_clusters_old);
            n_uni_clusters_new = length(uni_clusters_old);
            p_old = 1;
            p_new = 1;
            for jj = 1:length(uni_clusters_old)
                p_old = p_old*sum(chpt_cluster_id == uni_clusters_old(jj))/n_uni_clusters_old;
            end
            for jj = 1:length(uni_clusters_new)
                p_new = p_new*sum(chpt_cluster_id_temp == uni_clusters_new(jj))/n_uni_clusters_new;
            end
                           
            chpt_cluster_id;
            chpt_cluster_id_temp;
            psi_shift = p_new/p_old;
        end
    end
    
   
    
    
    %acceptance probability
    b = rand;
    acc_ratio = min([1,exp(llh_new-llh_old) * psi_shift]);
    
    %if accept
    cluster_pop_old = cluster_pop;
    if b < acc_ratio
        chpt_ind = chpt_ind_temp;
        chpt_cluster_id = chpt_cluster_id_temp;
        cluster_id = cluster_id_temp;
        %initialize as different functions
        for i = 1:ind
            temp_ind = max(find(i >= chpt_ind));
            cluster_id(i) = chpt_cluster_id(temp_ind);
            cluster_pop(cluster_id(i)) = cluster_pop(cluster_id(i))+1;
        end
        %relearn GPs
        %NOTE: currently inefficient, just need to rerun two clusters
        %also need to delete empty clusters
        num_clusters = max(cluster_id);
        first_pt = zeros(1,num_clusters);
        if num_clusters > length(cluster_pop)
            cluster_pop(end+1) =1;
        end
        clear gp_temp
        %GPs
        for i = 1:num_clusters
           gp_temp{i} = onlineGP.onlineGP_PE(params.bandwidth,params.noise,params.max_points,params.tol);  
        end
        try
        for i = 1:ind
            if first_pt(cluster_id(i)) == 0
                gp_temp{cluster_id(i)}.process(x(:,i),y(i));
                first_pt(cluster_id(i)) = 1;
            else
                gp_temp{cluster_id(i)}.update(x(:,i),y(i));
            end
        end
        catch
            2
        end
        
    end
    
    %convergence check
    if std(exp(llh_last/ind)) < 0.005*abs(mean(exp(llh_last/ind)))
        break;
    end
    llh_last = [llh_old,llh_last(1:end-1)];
   
   k
end %big loop


%posterior prediction
mu_predict = [];
for i = 1:num_chpt-1
    x0 = x(:,chpt_ind(i):chpt_ind(i+1)-1);
    y0 = y(chpt_ind(i):chpt_ind(i+1)-1);
    mu= gp_temp{chpt_cluster_id(i)}.predict(x0);
    mu_predict = [mu_predict, mu'];
end
x0 = x(:,chpt_ind(end):ind);
y0 = y(chpt_ind(end):ind);
mu= gp_temp{chpt_cluster_id(num_chpt)}.predict(x0);
mu_predict = [mu_predict, mu'];


out.params = params;
out.mean = mu_predict;
out.cluster_id = cluster_id;
out.chpt_ind = chpt_ind;

end
