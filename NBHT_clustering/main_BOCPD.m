%main_BOCPD 
%bayesian changepoint detection

close all;
clear all;


%generate data
ind = 200;
t = 1:ind;
x = [ones(1,ind/2)*0, ones(1,ind/2)];
x = x + randn(size(x))/10;
figure(1)
plot(t,x)



%initialize algorithm
mu_prior = 0;
var_prior = 1/100;
P_r = zeros(ind,ind); %probability of run length r. rows are runlength, column is time index
mu_stack = P_r;
var_stack = P_r;
P_rgx = P_r;
P_r(1,1) = 1;
P_rgx(1,1) = 1;
hazard = 100^-1; %0.001 is best


%compute mu_stack and sigma_stack before
for i = 2:ind
    i
    for j = 1:i
        mu_stack(j,i) = mean(x(i-j+1:i))*(length(x(i-j+1:i))-1)/length(x(i-j+1:i)) + 1*0;
        var_stack(j,i) = var(x(i-j+1:i))*(length(x(i-j+1:i))-1)/length(x(i-j+1:i)) + 1/length(x(i-j+1:i))*1/100;
    end
end
mu_stack(1,:) = mu_prior;
var_stack(1,:) = var_prior;

for i = 2:length(x)
   %observe data
   x0 = x(i);
   
   %eval predictive prob
   for r = i:-1:1
       %pi_t(r) = probabiity of x0 given parameters  
       pi_t(r) = normpdf(x0,mu_stack(r,i),var_stack(r,i)^0.5);
   end
   
   %growth probabilities
   for r = i:-1:2
      P_r(r,i) = P_r(r-1,i-1)*pi_t(r)*(1-hazard); 
   end
   %changepoint probabilities
   for r = i:-1:1
      P_r(1,i) = P_r(1,i) + P_r(r,i-1)*pi_t(r)*(hazard); 
   end
   
   %evidence
   evidence = sum(P_r(:,i));
   
   %posterior run length dist
   P_rgx(:,i) = P_r(:,i)/evidence;
   
   %posterior parameters
   %mu_stack(:,i) = mu_stack(:,i-1);
   %mu_stack(i,i) = mu_stack(i-1,i)*(i-1)/i + 1/i*x0 ;
   %sigma_stack(i,i) = sigma_stack(i-1,i)*(i-1)/i + 
  
   %prediction
   mu_predict(i) = sum(P_rgx(:,i).*mu_stack(:,i));
    
end
%%
%plot
%[XX YY] = meshgrid(1:100);
%X = XX(:);
%Y = YY(:);
%Z = P_rgx(:);
figure(2)
surf(P_rgx(1:200,1:200))
figure(1)
hold on
plot(mu_predict,'r');
%contour(P_rgx,100)
return;
%%
figure(2)
for i = 1:length(X)
    plot(X(i),Y(i), 'ok','markerfacecolor',[1 1 1]*(1-Z(i)))
    hold on
end


