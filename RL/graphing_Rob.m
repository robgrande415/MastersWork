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
close all;
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

%load new timer
load puddle_cpace
timerCpace = timer(timer>0);


x = size(ag_swap2,2);
fontsize = 20;
fh = figure;
%plot(1:x,cau,1:x,no_swap,1:x,swap,1:x,ag_swap);
%legend('cautious swap', 'no swap', 'swap','aggressive swap');

set(gca, 'colororder', [0.7,0.0,1.0;1.0,0,0], 'nextplot', 'replacechildren');
%plot(1:x,ag_swap,1:x,[ag_swap2 zeros(1,200)]);
plot(1:x,ag_swap2,'LineWidth',1.5);
%legend('GPQ+swap', 'Cpace');

xlabel('Episode','FontSize',fontsize);
ylabel('Steps to goal','FontSize',fontsize);
%title('5X5 Flat World');

figure;

%set(gca, 'colororder', [0.7,0.0,1.0;1.0,0,0], 'nextplot', 'replacechildren');

x = length(timerCpace);
%plot(1:x,timerSwap(1,1:x),1:x,timerCpace(1,1:x));
semilogy(1:x,timerCpace(1,1:x),'Color',[0.7,0.0,1.0]);
%legend('GPQ+swap', 'Cpace');
axis([0 x 1e-4 1e1])
xlabel('Step','FontSize',fontsize);
ylabel('CPU Time','FontSize',fontsize);
%title('Square Domain');


% DGPQ
load puddleswap_final
mean_Reward = mean(rews);
timerAll = timer;

ag_swap2 = mean_Reward;
ag_swap_dev2 = std_Dev_Reward;
timerDGPQ = zeros(1,size(timerAll,2));
for k=1:size(timer,2)
    timerDGPQ(1,k) = sum(timerAll(:,k)) / sum(timerAll(:,k) > 0);
end
x = size(ag_swap2,2);
figure(fh);
hold on
plot(1:x,ag_swap2,'LineWidth',1.5,'Color',[1,0,0]);
set(gca,'FontSize',fontsize)
gg=legend('CPACE','DGPQ');
set(gg,'FontSize',fontsize)
x = 5000;
figure(2);
hold on
semilogy(1:x,timerDGPQ(1,1:x),'Color',[1,0.0,0.0]);
set(gca,'FontSize',fontsize)
gg=legend('CPACE','DGPQ');
set(gg,'FontSize',fontsize)

g=figure(1);
h=figure(2);

%%
saveTightFigure(g,'../icml2014_gpq/Figures/square_steps.pdf')
saveTightFigure(h,'../icml2014_gpq/Figures/square_computation.pdf')

