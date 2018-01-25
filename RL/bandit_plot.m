close all
load banditno_swap
no_swap = rews;
no_swap = no_swap+1;
no_swap = no_swap*10;
no_swap = no_swap+1;

load banditswap
swap = rews;
swap = swap+1;
swap = swap*10;
swap = swap+1;

load bandit_epsgreedy
eps_greedy = train_Reward;
eps_greedy = 10*(eps_greedy+1)+1;

%% sd
close all
figure(1)
subplot(3,1,1)
plot(no_swap,'LineWidth',1)
axis([0 300 0.5 2.5])
gg=legend('Greedy GP')
set(gg,'FontSize',13)
title('Action taken for 2 Action MDP','FontSize',13)
%ylabel('Action Chosen','FontSize',13)

hold on
subplot(3,1,2)
plot(eps_greedy,'g','LineWidth',1)
axis([0 300 0.5 2.5])
gg=legend('\epsilon-Greedy GP','FontSize',13)
ylabel('Action Chosen','FontSize',13)
set(gg,'FontSize',13)

hold on
subplot(3,1,3)
plot(swap,'k','LineWidth',1)
axis([0 300 0.5 2.5])
gg= legend('DGPQ','FontSize',13)
set(gg,'FontSize',13)
%ylabel('Action Chosen','FontSize',13)

xlabel('Step','FontSize',13)
%axis([0 300 0 7])
h = figure(1);
