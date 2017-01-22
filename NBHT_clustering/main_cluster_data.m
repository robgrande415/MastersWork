%This is the main script that will generate data and then using clustering


close all; clear all;
addpath ../Auxiliary/
%get data
[x,y,act_model] = get_data('RobotDance1');
plot(x')
figure 
plot(y)
%%
% online GP stuff
%robot dance

params.bandwidth = 1.5; 
params.tol = 1e-4; 
params.noise = sqrt(0.2 );
params.parameter_est = 0;
params.max_points = 100;
params.detect_size = 20;
params.kl_tol = 0.5;
params.bin_tol = -0.5;


%airplace
%{
params.bandwidth = 0.3085; 
params.tol = 1e-6; 
params.noise = sqrt(2.04 );
params.parameter_est = 0;
params.max_points = 100;
params.detect_size = 30;
params.kl_tol = 100;
params.bin_tol = inf;
%}
%{
params.bandwidth = 1; 
params.tol = 1e-6; 
params.noise = sqrt(0.6 );
params.parameter_est = 0;
params.max_points = 2;
params.detect_size = 20;
params.kl_tol = 0.5;
params.bin_tol = 1;
%}
%learn hyperparameters
if 0
    gppe = onlineGP_PE(params.bandwidth,params.noise,params.max_points,params.tol);
    gppe.process(x(:,1),y(1,1));
    for i = 1:50
        gppe.update(x(:,i),y(1,i));
        gppe.update_param(x(:,i),y(1,i));
    end
    params.noise = gppe.get('snr')
    params.bandwidth = gppe.get('sigma')
end
%%
out = run_cluster_data(x,y,params,1);

%%
if 1
    temp = ones(1,1000);
    act_model = [temp, temp*2, temp, temp*3, temp, temp*4, temp*2, temp*3, temp];
    %act_model = [ones(1,1500),ones(1,1500)*2];
    
end
est_model_fixed = out.est_model;
est_model_fixed(2044:2125) = 1;
est_model_fixed(4049:4130) = 1;
est_model_fixed(6096:6177) = 2;
est_model_fixed(7298:7379) = 3;
est_model_fixed(8049:8130) = 1;

t =1:length(act_model);
t = t/5;
figure(3)
plot(t,out.est_model,'-.','Linewidth',2)
hold on
plot(t,act_model,'r','Linewidth',2)
legend('Estimated Model','Actual Model')
xlabel('Time (s)')
ylabel('Model number')
title('Model Identification for Robot Interatction')
set(findobj(figure(3),'Type','Text'),'FontSize',12)
return;
%plot(est_model_fixed,'--g')
subplot(1,2,2)
plot(real(out.kl_vals))

figure(2)
subplot(1,2,1)
plot(y)
hold on
plot(out.mean,'r')
subplot(1,2,2)
plot(act_model(1:end-1)-out.mean)
mean(abs(act_model(1500:end-1)-out.mean(1500:end)))
std(abs(act_model(1500:end-1)-out.mean(1500:end)))

