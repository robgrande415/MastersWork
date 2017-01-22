function out = run_BOCPD(x,y,params,predict_on)

%main_BOCPD 
%bayesian changepoint detection
%generate data
ind = length(x);
window = min(ind+1,2000);

%initialize algorithm
mu_prior = 0;
var_prior = params.noise^2;
P_r = zeros(ind+1,ind); %probability of run length r. rows are runlength, column is time index
mu_stack = P_r;
var_stack = P_r+1+params.noise^2;
P_rgx = P_r;
P_r(1,1) = 1;
P_rgx(1,1) = 1;
hazard = params.kl_tol; %0.001 is best
mu_stack_predict = zeros(1,ind);
sigma_stack_predict = zeros(1,ind);
%window = 1000;
%load data
%compute mu_stack and sigma_stack before
if 1
    tic
    for i = 2:ind
        gp_temp = onlineGP.onlineGP_PE(params.bandwidth,params.noise,params.max_points,params.tol);
        for j = 1:min([i,window])
            if j ==1
                gp_temp.process(x(i),y(i));
            else
                gp_temp.update(x(i-j+1),y(i-j+1))
            end
            %gp_temp.train(x(i-j+1:i),y(i-j+1:i));
            [f,var] = gp_temp.predict(x(i));
            mu_stack(j+1,i) = f;
            var_stack(j+1,i) = var+params.noise^2;
        end
        if mod(i,10) == 0
            i
        end
    end
    toc
    mu_stack(1,:) = mu_prior;
    var_stack(1,:) = var_prior;
    save data mu_stack var_stack
end   

for i = 2:length(x)
   %observe data
   y0 = y(i);
   
   %eval predictive prob
   pi_t = zeros(1,i);
   for r = i:-1:1
       %pi_t(r) = probabiity of x0 given parameters  
       pi_t(r) = normpdf(y0,mu_stack(r,i),var_stack(r,i)^0.5);
   end
   
   %growth probabilities
   for r = i:-1:2
      P_r(r,i) = P_r(r-1,i-1)*pi_t(r)*(1-hazard); 
   end
   %changepoint probabilities
   P_r(1,i) = P_r(1:i,i-1)'*(pi_t(1:i))'*hazard;
   %for r = i:-1:1
   %   P_r(1,i) = P_r(1,i) + P_r(r,i-1)*pi_t(r)*(hazard); 
   %end
   
   %evidence
   evidence = sum(P_r(:,i));
   if evidence ==0
       i
       evidence
       log(P_r(1:i,i))
       out.P_r = P_r;
       out.pi_t = pi_t;
       out.mean = mu_stack_predict;
       return;
   end
   %posterior run length dist
   P_rgx(:,i) = P_r(:,i)/evidence;
   P_r(:,i) = P_r(:,i)/evidence;
   %posterior parameters
   %mu_stack(:,i) = mu_stack(:,i-1);
  
   %prediction
   mu_stack_predict(i) = P_rgx(:,i)'*mu_stack(:,i);
   sigma_stack_predict(i) = P_rgx(:,i)'*var_stack(:,i);
end

out.P_rgx = P_rgx;
out.params = params;
out.mean = mu_stack_predict;
out.stdev = sigma_stack_predict;
out.P_r = P_r;
out.pi_t = pi_t;
out.mu_stack = mu_stack;
end

%%
%plot
%[XX YY] = meshgrid(1:100);
%X = XX(:);
%Y = YY(:);
%Z = P_rgx(:);
%figure(2)
%surf(P_rgx(500:600,500:600))
%contour(P_rgx,100)
%%
%figure(2)
%for i = 1:length(X)
%    plot(X(i),Y(i), 'ok','markerfacecolor',[1 1 1]*(1-Z(i)))
%    hold on
%end
%{

%}

