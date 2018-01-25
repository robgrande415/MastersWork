%load 'acrobotno_swap.mat'
%no_swap = rews; %mean_Reward;
%no_swap_dev = std_Dev_Reward;

%load 'swap.mat'
%swap = mean_Reward;
%swap_dev = std_Dev_Reward;


%load 'cautious_swap.mat'
%cau = mean_Reward;
%cau_dev = std_Dev_Reward;


%load 'wrightData/grid10_swap.mat'
%ag_swap = mean_Reward;
%ag_swap_dev = std_Dev_Reward;
%timerSwap = mean(timer);

rewardAll = zeros(10,200);
timerAll = zeros(10,20000);

load 'puddle_cpace_big01Noise_etol05_Goal15_3Runs_good.mat' 
rewardAll(1:3,:) = rews;
timerAll(1:3,:) = timer;
load 'puddle_cpace_big01Noise_etol05_Goal15_3More.mat' 
rewardAll(4:6,:) = rews;
timerAll(4:6,:) = timer;
load 'puddle_cpace_big01Noise_etol05_Goal15_2Runs.mat' 
rewardAll(7:8,:) = rews;
timerAll(7:8,:) = timer;
load 'puddle_cpace_big01Noise_etol05_Goal15_2More.mat' 
rewardAll(9:10,:) = rews;
timerAll(9:10,:) = timer;

mean_Reward = mean(rewardAll);

ag_swap2 = mean_Reward;
ag_swap_dev2 = std_Dev_Reward;
timerCpace = zeros(1,size(timerAll,2));
for k=1:size(timer,2)
    timerCpace(1,k) = sum(timerAll(:,k)) / sum(timerAll(:,k) > 0);
end
x = size(ag_swap2,2);

fh = figure;
%plot(1:x,cau,1:x,no_swap,1:x,swap,1:x,ag_swap);
%legend('cautious swap', 'no swap', 'swap','aggressive swap');

set(gca, 'colororder', [0.7,0.0,1.0;1.0,0,0], 'nextplot', 'replacechildren');
%plot(1:x,ag_swap,1:x,[ag_swap2 zeros(1,200)]);
plot(1:x,ag_swap2);
%legend('GPQ+swap', 'Cpace');
legend('Cpace');

xlabel('episode');
ylabel('steps to goal');
%title('5X5 Flat World');

figure;

set(gca, 'colororder', [0.7,0.0,1.0;1.0,0,0], 'nextplot', 'replacechildren');

x = 5000;
%plot(1:x,timerSwap(1,1:x),1:x,timerCpace(1,1:x));
semilogy(1:x,timerCpace(1,1:x));
%legend('GPQ+swap', 'Cpace');
legend('Cpace');
axis([0 5000 1e-5 1e1])
xlabel('time-step');
ylabel('CPU Time');
%title('Square Domain');



%print( fh, 'out.pdf', '-dpdf', '-r600');

