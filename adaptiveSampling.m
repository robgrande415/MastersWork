%this script will duplicate the results of Singh in the adaptive sensing
%paper. Then it will expand the results and perform GP fitting to the
%regions of interest

%1-d
close all
clear;
for i = 1:5
    a(i) = 1/i*randn;
end
x = 0:0.01:4;
f = 0*x;

%for "continuous" step
stepLoc = mean(x) + 0.5;
f = 1./(1+exp(-50*(x-mean(x))));
f = smooth(f,25);
plot(x,f)
hold on
%add noise
snr = 0.10;

%n/2 sampled points
n = 100;
xData = 0:4/(n-1):4;
y = 1./(1+exp(-50*(xData-mean(x)))) + snr*randn(size(xData));
plot(xData,y,'x')