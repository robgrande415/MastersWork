% Rob Grande
% this script will compare results for NBHT, CRP/constant alpha, BOCPD, and DPGP

%% generate data
clear;
close all;
addpath ../Auxiliary/
addpath ../
%flags
plot_me = 1;
NBHT_flag = 1;
CRP_flag = 0;
BOCPD_flag = 0;
DPGP_flag = 0;
MCMC_flag = 0;
FORGET_flag = 0;
%get data
filename = 'Parabola';
num_trials = 100;
for j = 1:num_trials
    j
    close all;
    [x,y,f,act_model] = get_data(filename);
    %plot(x')
    %figure 
    %plot(f)
    %plot(x,f,'x')
    %plot(x,y,'rx')


    %% run NHBT

    if NBHT_flag
        params = get_params(filename,'NBHT');
        kl_tol_settings = params.kl_tol;
        kl_tol_settings = 0.6;
        kl_tol_settings = linspace(0.1,2,20);
        detect_settings = linspace(3,50,20);
        kl_tol_settings = repmat(kl_tol_settings,1,20);
        detect_settings = kron(detect_settings,ones(1,20));
        detect_settings = round(detect_settings);
        %params.detect_size = 30;
        %params.bin_tol = +inf;
        %learn hyperparameters
        if 0
            gppe = onlineGP_PE(params.bandwidth,params.noise,params.max_points,params.tol);
            gppe.process(x(:,1),y(1,1));
            for i = 1:100
                gppe.update(x(:,i),y(1,i));
                gppe.update_param(x(:,i),y(1,i));
            end
            params.noise = gppe.get('snr')
            params.bandwidth = gppe.get('sigma')
        end
        tic
        for i = 1:length(kl_tol_settings)
            params.kl_tol = kl_tol_settings(i);
            params.detect_size = detect_settings(i);
            NBHT{i,j} = run_cluster_data(x,y,params,1);
            NBHT{i,j}.time = toc;
            NBHT{i,j}.act_model = act_model;
            NBHT{i,j}.f =f;
        end
    end 
    
    %% forgetting kernel
    if FORGET_flag
        params = get_params(filename,'NBHT');
        kl_tol_settings = params.kl_tol;
        %learn hyperparameters
        if 0
            gppe = onlineGP_PE(params.bandwidth,params.noise,params.max_points,params.tol);
            gppe.process(x(:,1),y(1,1));
            for i = 1:100
                gppe.update(x(:,i),y(1,i));
                gppe.update_param(x(:,i),y(1,i));
            end
            params.noise = gppe.get('snr')
            params.bandwidth = gppe.get('sigma')
        end
        tic
        for i = 1:length(kl_tol_settings)
            params.kl_tol = kl_tol_settings(i);
            FORGET{i,j} = run_GP_forget(x,y,params,1);
            FORGET{i,j}.time = toc;
            FORGET{i,j}.act_model = act_model;
            FORGET{i,j}.f =f;
        end
    end 
    
    %% CRP
    kl_tol_settings = [0.1 0.01];
    if CRP_flag
        params = get_params(filename,'CRP');
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

        for i = 1:length(kl_tol_settings)
            params.kl_tol = (1-kl_tol_settings(i))/kl_tol_settings(i);
            tic
            CRP{i,j} = run_cluster_data(x,y,params,1);
            CRP{i,j}.time = toc
            CRP{i,j}.act_model = act_model
            CRP{i,j}.f =f;
        end
    end 

    %% BOCPD

    kl_tol_settings = [0.01];
    if BOCPD_flag
        params = get_params(filename,'CRP');
        params.noise = sqrt(0.1);
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

        for i = 1:length(kl_tol_settings)
            params.kl_tol = kl_tol_settings(i);
            tic
            BOCPD{i,j} = run_BOCPD(x,y,params,1);
            BOCPD{i,j}.time = toc
            BOCPD{i,j}.act_model = act_model
            BOCPD{i,j}.f =f;
        end
    end 

    %% DPGP

    kl_tol_settings = ones(1,1)*[0.1];
    if DPGP_flag
        params = get_params(filename,'CRP');
        params.noise = sqrt(0.1);
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

        for i = 1:length(kl_tol_settings)
            params.kl_tol = kl_tol_settings(i);
            tic
            DPGP{i,j} = run_DPGP(x,y,params,1);
            DPGP{i,j}.time = toc;
            DPGP{i,j}.act_model = act_model;
            DPGP{i,j}.f =f;
            DPGP{i,j}.x = x;
            DPGP{i,j}.y = y;
        end
    end 

    %% DPGP-MCMC

    kl_tol_settings = ones(1,1)*[1];
    if MCMC_flag
        params = get_params(filename,'CRP');
        params.noise = sqrt(0.1);
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

        for i = 1:length(kl_tol_settings)
            params.kl_tol = kl_tol_settings(i);
            tic
            MCMC{i,j} = run_DPGP_MCMC(x,y,params,1);
            MCMC{i,j}.time = toc;
            MCMC{i,j}.act_model = act_model;
            MCMC{i,j}.f = f;
        end
    end 
end
%% do analysis

compare_analysis
%scratch_plots

return

%%
close all
figure
hold on
xp = linspace(-1,1,400);
fp1 = xp.^2;
fp2 = 2-xp.^2;
fp1 = xp.^2;
fp2 = +0.5+0.5*xp.^2;
yp1 = fp1+randn(size(xp))*0.2;
yp2 = fp2+randn(size(xp))*0.2;
plot(xp,fp1,'LineWidth',3)
plot(xp,fp2,'g','LineWidth',3)
legend('f_1(x)','f_2(x)')
plot(xp,yp1,'x','LineWidth',2,'MarkerSize',10)
plot(xp,yp2,'gx','LineWidth',2,'MarkerSize',10)
xlabel('x')
ylabel('y')




