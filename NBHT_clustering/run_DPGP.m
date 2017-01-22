function out = run_DPGP(x,y,params,predict_on)

%main_DPGP 
ind = length(x);
alpha = params.kl_tol;
alpha = 0.01;
%init
num_clusters = 2;
cluster_id = randi(num_clusters,1,ind);
cluster_pop = zeros(1,num_clusters);
first_pt = zeros(1,num_clusters);

%GPs
for i = 1:num_clusters
   gp_temp{i} = onlineGP.onlineGP_PE(params.bandwidth,params.noise,params.max_points,params.tol);  
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


%get data prob
llh_data = zeros(1,num_clusters);
for i = 1:num_clusters
    if cluster_pop(i) > 0
        x_ind = cluster_id==i;
        y_cl = y(x_ind);
        [mu, var] = gp_temp{i}.predict(x(x_ind));
        llh_data(i) = sum(log(normpdf(y_cl',mu,var'+params.noise^2)));
    end
end
pr_data = log((cluster_pop)/(ind-1+alpha));
data_Post_old = sum(pr_data + llh_data);
max_data_Post = data_Post_old;
out = get_MAP(x,y,params,cluster_id,gp_temp);
num_clusters = numel(unique(cluster_id));
cluster_id_list = unique(cluster_id);



%GIBBS
%loop and sample
MAX_ITER = 10;
for k = 1:MAX_ITER
    for j = 1:ind
        %sample pt
        ss_ind = j
        x0 = x(:,ss_ind);
        y0 = y(ss_ind);
        cluster_id_rem = cluster_id(ss_ind);
        %get likelihood of curr clusters
        llh = zeros(1,num_clusters+1);
        prior = llh;
        mu_stack = llh;

        for i = 1:num_clusters
            if cluster_pop(i) > 0
                [mu, var] = gp_temp{i}.predict(x0);
                mu_stack(i) = mu;
                llh(i) = log(normpdf(y0,mu,var+params.noise^2));
                
                %experts have equal weight
                if i == cluster_id_rem
                    jitter = 1e-5;
                    prior(i) = log((cluster_pop(i)-1+jitter)/(ind-1+alpha));
                else
                    prior(i) = log((cluster_pop(i))/(ind-1+alpha));
                end
                
                %gating function
                %temp = find(cluster_id == i);
                %temp_pts = x(temp);
                %prior(i) = sum(exp(-(temp-j).^2/200^2));
                %equal weight
                %prior(i) = numel(temp);
                
            end
        end


        %get LLH f new points
        mu = y0*1/(1+params.noise^2);
        var = (1 + 1/(params.noise^2))^-1;
        llh(end) = log(normpdf(y0,mu,var+params.noise^2));
        %no gate
        prior(end) = log(alpha/(ind-1+alpha));
        %prior(end) = alpha;
        %gate
        %prior(end) = 1;
        %normalize prior
        %prior  = prior/(sum(prior));
        %prior = log(prior);
        
        MAP = llh + prior;
        %hard clustering
        %[~,new_id] = max(MAP);
        old_id = cluster_id(ss_ind);
        %cluster_id(ss_ind) = new_id;
        
        
        %EDIT BY ROB: DON'T DO HARD CLUSTERING
        %get best cluster
        %[t,new_id] = max(MAP);
        MAP = exp(MAP)/sum(exp(MAP)); %normalize
        %create cascading effect
        for ii = 2:length(MAP)
            MAP(ii) = MAP(ii) + MAP(ii-1);
        end
        b = rand;
        new_id = find(MAP>b,1,'first');
        cluster_id(ss_ind) = new_id;
        
        if ~isempty(find(cluster_id_list == new_id, 1))
            cluster_pop(new_id) = cluster_pop(new_id)+1;
            cluster_pop(old_id) = cluster_pop(old_id)-1;
            if cluster_pop(old_id) == 0
                cluster_pop(old_id) = [];
                cluster_id(cluster_id > old_id) = cluster_id(cluster_id > old_id)-1;
                cluster_id_list(cluster_id_list>old_id) = cluster_id_list(cluster_id_list>old_id)-1;
            end
        else
            cluster_pop(end+1) = 1;
            cluster_pop(old_id) = cluster_pop(old_id)-1;
            if cluster_pop(old_id) == 0
                cluster_pop(old_id) = [];
                cluster_id(cluster_id > old_id) = cluster_id(cluster_id > old_id)-1;
                cluster_id_list(cluster_id_list>old_id) = cluster_id_list(cluster_id_list>old_id)-1;
            
            end
        end
    
      
        
        
        %relearn GPs
        %NOTE: currently inefficient, just need to rerun two clusters
        %also need to delete empty clusters
        num_clusters = numel(unique(cluster_id));
        cluster_id_list = unique(cluster_id);
        first_pt = zeros(1,num_clusters);
        
        clear gp_temp
        %GPs
        for i = 1:num_clusters
           gp_temp{cluster_id_list(i)} = onlineGP.onlineGP_PE(params.bandwidth,params.noise,params.max_points,params.tol);  
        end
        
        cluster_pop_old = cluster_pop;
        
        
        for i = 1:ind
            if first_pt(cluster_id(i)) == 0
                gp_temp{cluster_id(i)}.process(x(:,i),y(i));
                first_pt(cluster_id(i)) = 1;
            else
                gp_temp{cluster_id(i)}.update(x(:,i),y(i));
            end
        end
        
        %{
        for i = 1:num_clusters
            x_ind = cluster_id==cluster_id_list(i);
            y_cl = y(x_ind);
            gp_temp{i}.process(x(x_ind),y_cl);
        end
        %}
        %create new cluster in interim
        %{
        if max(cluster_id) > num_clusters
            gp_temp{max(cluster_id)} = onlineGP.onlineGP_PE(params.bandwidth,params.noise,params.max_points,params.tol); 
            cluster_pop(max(cluster_id)) = 1;
            
        end
        num_clusters = max(cluster_id);
        %}
        

    end %re assign
    k
   
    

    %convergence check: llh
    %get data prob
    llh_data = zeros(1,length(cluster_pop));
    for i = 1:num_clusters
        
            x_ind = cluster_id==cluster_id_list(i);
            y_cl = y(x_ind);
            [mu, var] = gp_temp{i}.predict(x(x_ind));
            llh_data(i) = sum(log(normpdf(y_cl',mu,var'+params.noise^2)));
    end
    cluster_pop
    llh_data(llh_data==0) = [];
    pr_data = log((cluster_pop(cluster_pop>0))/(ind-1+alpha))
    llh_data
    try
        data_Post = sum(pr_data + llh_data);
    catch
        return;
    end
    if data_Post >= max_data_Post
        out = get_MAP(x,y,params,cluster_id,gp_temp);
    end
        
    data_Post
    data_Post_old
    if abs(exp(data_Post)-exp(data_Post_old))/exp(data_Post_old) < 0.05
        return;
    end
    data_Post_old = data_Post;
    
     %convergence check: cluster populations
    %if sum(abs(cluster_pop_old -cluster_pop)) < 0.01*ind
        
       %break; 
    %end
    
    
    %plot for debugging
    figure(1)
    clf;
    colors = {'bx','go','r+','kx','co','y+'};
    for i = 1:num_clusters
        if cluster_pop(i) > 0
            x_ind = cluster_id==cluster_id_list(i);
            y_cl = y(x_ind);
            hold on
            plot(x(x_ind),y_cl,colors{mod(i-1,6)+1})
        end
    end
    pause(0.01);
end %big loop


end

function out = get_MAP(x,y,params,cluster_id,gp_temp)
    ind = size(x,2);
    mu_predict = zeros(1,ind);
    %do posterior prediction
    for ii = 1:ind
        x0 = x(:,ii);
        mu_predict(ii) = gp_temp{cluster_id(ii)}.predict(x0);
    end

out.params = params;
out.mean = mu_predict;
out.cluster_id = cluster_id;

end

