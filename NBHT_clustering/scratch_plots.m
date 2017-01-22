% plots

fgt = zeros(1,400);
nb = fgt;
for i = 1:size(FORGET,2)
    fgt = fgt+abs(FORGET{i}.mean-FORGET{i}.f);
    nb = nb+abs(NBHT{i}.mean-NBHT{i}.f);
end
fgt = fgt/size(FORGET,2);
nb = nb/size(FORGET,2);
close all;
plot(fgt,'LineWidth',1.5)
hold on;
plot(nb,'k','LineWidth',1.5)
axis([0 199 0 0.8])
h=legend('GP-Sliding Window','GP-NBC');
set(h,'FontSize',13);
xlabel('Time','FontSize',13)
ylabel('Averaged Error over 10 Trials','FontSize',13)
title('Averaged Error vs. Time','FontSize',13)

return;

%%
xplot = linspace(-1,1,500);
f1 = xplot.^2;
f2 = 2-xplot.^2;
y1 = f1 + randn(size(f1))*0.2;
y2 = f2 + randn(size(f1))*0.2;

figure

plot(xplot,xplot.^2,'b','LineWidth',3)
hold on
plot(xplot,2-xplot.^2,'g','LineWidth',3)
plot(xplot,y1,'bx');
plot(xplot,y2,'gx')
xlabel('x')
ylabel('y')
legend('f_1','f_2')
