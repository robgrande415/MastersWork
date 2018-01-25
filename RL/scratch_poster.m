% plot for poster
close all;
clear all;
lw = 2;
fs = 14;
%create q function
xplot = linspace(0,1,1000);
Qplot = sin(xplot*pi*2)-1;
y = Qplot +randn(size(Qplot));

%train Q function
params.N_act=1; params.rbf_mu=0.25; params.sigma=0.1;params.N_budget=100; params.tol=1e-4; params.A = 1; params.Lip = 2*pi;
gpr = onlineGP.onlineGP_MA(params.N_act,params.rbf_mu,params.sigma,params.N_budget,params.tol,params.A);
avgr = onlineGP.onlineAVGR_MA(params.N_act,params.rbf_mu/4,params.sigma,params.N_budget,params.tol,params.Lip);

xtrain = rand(1,10);
xtrain = [xtrain,0.25+rand(1,10)/10];
Qtrain = sin(xtrain*pi*2)-1;
ytrain = Qtrain + randn(size(Qtrain))/5;

for i = 1:length(xtrain)
   gpr.update(xtrain(i),1,ytrain(i));
    
    
end

%get prediction
[gpout,v] = gpr.predict(xplot,1);
H=confplot(xplot,gpout,sqrt(diag(v)));
set(H(1),'LineWidth',lw)
hold on
plot(xplot,Qplot,'r','LineWidth',lw)

%q function
xt = 0.35;
yt = gpr.predict(0.35,1)+0.03;
avgr.update(xt,1,yt)
avgr.update(xt,1,yt)


%get prediction
for i = 1:1000
qout(i) = avgr.getMax(xplot(i));
end
plot(xplot,qout,'g','LineWidth',lw)
title('Adding First Known Point','FontSize',fs)
xlabel('X','FontSize',14)
ylabel('Q(x)','FontSize',14)
%%
h = figure(1);
saveTightFigure(h,'FirstIteration.pdf')



%% more detailed simulation


% plot for poster
close all;
clear all;
lw = 1.8;
fs = 14;
%create q function
xplot = linspace(0,1,1000);
Qplot = sin(xplot*pi*2)-1;
y = Qplot +randn(size(Qplot));

%train Q function
params.N_act=1; params.rbf_mu=0.25; params.sigma=0.1;params.N_budget=100; params.tol=1e-4; params.A = 1; params.Lip = 2*pi;
gpr = onlineGP.onlineGP_MA(params.N_act,params.rbf_mu,params.sigma,params.N_budget,params.tol,params.A);
avgr = onlineGP.onlineAVGR_MA(params.N_act,params.rbf_mu/4,params.sigma,params.N_budget,params.tol,params.Lip);

xtrain = rand(1,10);
xtrain = [xtrain,0.7+rand(1,10)/20];
Qtrain = sin(xtrain*pi*2)-1;
ytrain = Qtrain + randn(size(Qtrain))/10;

for i = 1:length(xtrain)
   gpr.update(xtrain(i),1,ytrain(i));
    
    
end

%get prediction
[gpout,v] = gpr.predict(xplot,1);
H=confplot(xplot,gpout,sqrt(diag(v)));
set(H(1),'LineWidth',lw)
hold on
plot(xplot,Qplot,'r','LineWidth',lw)

%q function
xt = 0.35;
yt = gpr.predict(0.35,1)+0.01;
avgr.update([xt;0],1,yt)

%add random points
for j = 1:20
    xt = rand;
    yt = 1/2*avgr.getMax([xt;0]) + 1/2*(sin(xt*2*pi)-1);
    avgr.update([xt;0],1,yt);
end

%get prediction
for i = 1:1000
qout(i) = avgr.predict([xplot(i);0],1);
end
plot(xplot,qout,'g','LineWidth',lw)
title('Example Representation of Q(x)','FontSize',fs)
xlabel('X','FontSize',14)
ylabel('Q(x)','FontSize',14)
%%
h = figure(1);
saveTightFigure(h,'FinishedIteration.pdf')
