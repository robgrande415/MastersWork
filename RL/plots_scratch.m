% plots
close all
load rews_f16
fontsize = 16;
%h = confplot(1:300,mean_Reward,std_Dev_Reward,std_Dev_Reward);
h = plot(1:300,mean_Reward,std_Dev_Reward,std_Dev_Reward);
set(gca,'FontSize',fontsize)
set(h(1),'LineWidth',1)
set(h(1),'Color','r')
%set(h(2),'FaceColor',[0.8 0.8 1])
title('Reward per Episode','FontSize',fontsize)
xlabel('Episode','FontSize',fontsize)
ylabel('Reward Averaged over 10 Trials','FontSize',fontsize)

load f16swap_lip10
mean_Reward = mean(rews);
std_Dev_Reward = std(rews);
hold on
plot(1:1000,mean_Reward,'g','LineWidth',1);
axis([0 1000 -700 0])
gg = legend('L_Q = 5','L_Q = 10','Location','SouthEast');
set(gg,'FontSize', fontsize)
%{
max_rew = max(rews);
hold on
plot(max_rew,'g')
min_rew = min(rews);

plot(min_rew,'r')
%}


saveTightFigure(g, '../icml2014_gpq/Figures/Rews_f16.pdf')

%%