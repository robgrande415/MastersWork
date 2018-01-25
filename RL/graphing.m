load 'gridno_swap.mat'
no_swap = rews; %mean_Reward;
no_swap_dev = std_Dev_Reward;

%load 'gridswap.mat'
%swap = rews;
%swap_dev = std_Dev_Reward;


load 'gridcautious_swap.mat'
cau = rews;
cau_dev = std_Dev_Reward;


%load 'gridcautious_swap.mat'
%ag_swap = rews; %mean_Reward;
%ag_swap_dev = std_Dev_Reward;


x = size(no_swap,2);

figure;
plot(1:x,cau,1:x,no_swap,1:x,swap);
legend('cautious swap', 'no swap', 'swap','aggressive swap');



xlabel('episode');
ylabel('reward');
title('Acrobot')