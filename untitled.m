% scratch for CDC talk

clear all;
close all;

x = linspace(0,2,200);
f = -1/2*x.^2 + x;
y = f + randn(size(f))*0.2;

%train 3 GPs
gpr1 = onlineGP.onlineGP(0.05,0.2,100,0.01);
gpr1.process(x(1),y(1));
gpr2 = onlineGP.onlineGP(1,0.2,100,0.01);
gpr2.process(x(1),y(1));
gpr3 = onlineGP.onlineGP(6,0.2,100,0.01);
gpr3.process(x(1),y(1));
for i = 1:length(x)
   gpr1.update(x(i),y(i));
   gpr2.update(x(i),y(i));
   gpr3.update(x(i),y(i));
end

mu1 = gpr1.predict(x);
mu2 = gpr2.predict(x);
mu3 = gpr3.predict(x);
figure
hold on
plot(x,y,'xk','MarkerSize',10,'LineWidth',2)
plot(x,f,'k','LineWidth',4)
plot(x,mu1,'b','LineWidth',4)
plot(x,mu2,'g','LineWidth',4)
plot(x,mu3,'r','LineWidth',4)
hh = legend('Data','Ground Truth','\theta = 0.05','\theta = 1','\theta = 6') ;
set(hh,'FontSize',15)
xlabel('X (input)','FontSize',15)
ylabel('y (observations)','FontSize',15)
h = figure(1);
saveTightFigure(h, 'CompHP.pdf')

%%

close all;
clear all;
x = linspace(0,5,1000);
f = (0.33*sin(x) + 0.7*cos(2.4*x))*1;
y = f + randn(size(f))*0.1;

gpr2 = onlineGP.onlineGP(1,0.2,100,0.01);
gpr2.process(x(1),y(1));
for i = 1:length(x)
   gpr2.update(x(i),y(i));
end

mu1 = gpr2.predict(x);
figure
hold on
plot(x,y,'xk','MarkerSize',10,'LineWidth',1.5)
plot(x,f,'g','LineWidth',4)
plot(x,mu1,'r','LineWidth',4)

BV = gpr2.get('BV');
mu_BV = gpr2.predict(BV);
plot(BV,mu_BV,'ro','MarkerSize',25,'LineWidth',4)
hh = legend('Data','Ground Truth','GP Prediction','Basis Vectors');
set(hh,'FontSize',20)
xlabel('X (input)','FontSize',15)
ylabel('y (observations)','FontSize',15)
h = figure(1);
saveTightFigure(h, 'PIGP.pdf')






