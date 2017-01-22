%this script trains a gp given some function and noisy data. It has both 1d
%and 2d

close all
clear;
for i = 1:5
    a(i) = 1/i*randn;
end
x = 0:0.05:2;
f = 0*x;
for i = 1:length(a)
   f = f + a(i)*cos(i*x+randn); 
end

%for step
%f(1:round(length(x)/2)) = 0;
%f(round(length(x)/2)+1:end) = 1;
plot(x,f)
hold on
%add noise
snr = 0.15;
y = f + snr*randn(size(f));
plot(x,y,'x')

indices = 1:round(length(x)/2);
for shift = 0:5:length(x)-max(indices)
    tic;
    xsubset = x(indices+shift);
    ysubset = y(indices+shift);
    param = minLG(xsubset,ysubset,[1 2 0.2],1000,0);
    toc
    A = param(1);
    s = param(2);
    snr = param(3);
    param = param(1:end-1);
    for i = 1:length(xsubset)
        for j = 1:length(xsubset)
           Suu(i,j) = kernel(xsubset(i),xsubset(j),param,snr);
        end
    end
    for i = 1:length(xsubset)
        for j = 1:length(xsubset)
           Syu(i,j) = kernel(xsubset(i),xsubset(j),param,0); 
        end
    end
    yhat = Syu*Suu^-1*ysubset';
    sigmax = Syu - Syu*Suu^-1 * Syu';
    sxx = diag(sigmax);
    theoryStDev = sqrt(mean(sxx))
    %variance = sum((yhat-f(indices+shift)).^2)/length(x);
    %experimentalStDev = sqrt(variance)
    errorbar(xsubset,yhat,0.5*sxx.^0.5,'r')
    param
end

indices = 1:round(length(x));
for shift = 0:5:length(x)-max(indices)
    tic;
    xsubset = x(indices+shift);
    ysubset = y(indices+shift);
    param = minLG(xsubset,ysubset,[1 2 0.2],1000,0);
    toc
    A = param(1);
    s = param(2);
    snr = param(3);
    param = param(1:end-1);
    for i = 1:length(xsubset)
        for j = 1:length(xsubset)
           Suu(i,j) = kernel(xsubset(i),xsubset(j),param,snr);
        end
    end
    for i = 1:length(xsubset)
        for j = 1:length(xsubset)
           Syu(i,j) = kernel(xsubset(i),xsubset(j),param,0); 
        end
    end
    yhat = Syu*Suu^-1*ysubset';
    sigmax = Syu - Syu*Suu^-1 * Syu';
    sxx = diag(sigmax);
    theoryStDev = sqrt(mean(sxx))
    %variance = sum((yhat-f(indices+shift)).^2)/length(x);
    %experimentalStDev = sqrt(variance)
    errorbar(xsubset,yhat,0.5*sxx.^0.5,'g')
    param
end


%% 2-d
close all
clear;
for i = 1:5
    a(:,i) = 1/i*randn(3,1);
end
x1 = 0:0.4:4;
x2 = x1;
f = zeros(length(x1));
for i = 1:length(a)
    for j = 1:length(x1)
        for k = 1:length(x2)
            f(j,k) = f(j,k) + a(1,i)*cos(i*x1(j)) + a(2,i)*cos(i*x2(k)) + a(3,i)*cos(i*sqrt(x2(k)*x1(j)) ); 
        end
    end
end
[mesh1 mesh2] = meshgrid(x1,x2);
%for step
%f(1:round(length(x)/2)) = 0;
%f(round(length(x)/2)+1:end) = 1;
h = surf(mesh1,mesh2,f,'EdgeColor','Black');
set(h,'FaceColor',[0 0 1],'FaceAlpha',0.5);
hold on
%add noise
snr = 0.25;
y = f + snr*randn(size(f));
plot3(mesh1,mesh2,y,'x')

%creates set of all data points
xvec = kron(x1,[ones(1,length(x1)); zeros(1,length(x1))]) + ...
            kron([zeros(1,length(x1)); ones(1,length(x1))],x2);
data = y(:)';
param = fminsearch(@(param) lg(xvec,data,param,snr),[1 2]);

A = param(1);
s = param(2);
for i = 1:length(xvec)
    for j = 1:length(xvec)
       Suu(i,j) = kernel(xvec(:,i),xvec(:,j),param,snr);
    end
end
for i = 1:length(xvec)
    for j = 1:length(xvec)
       Syu(i,j) = kernel(xvec(:,i),xvec(:,j),param,0);
    end
end
yhat = Syu*Suu^-1*data';
sigmax = Syu - Syu*Suu^-1 * Syu';
sxx = diag(sigmax);
yhatMat = vec2mat(yhat,length(x1))';
h = surf(mesh1,mesh2,yhatMat);
set(h,'FaceColor',[1 0 0],'FaceAlpha',0.5);
variance = sum((yhat-f(:)).^2)/length(data)
stdev = sqrt(variance)
