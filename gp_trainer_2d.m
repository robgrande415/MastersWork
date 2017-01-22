close all
clear;
for i = 1:5
    a(i) = 1/i*randn;
end
n = 10;
b1 = 0;
b2 = 4;
x = b1:(b2-b1)/(n-1):b2;
range = x;
x = kron([ones(1,length(x)); zeros(1,length(x))],x) + ...
        kron(x,flipud([ones(1,length(x)); zeros(1,length(x))]));
f = zeros(1,length(x));


for i = 1:length(a)
   %f = f + a(i)*cos(i*x); 
end

%for sigmoid
f = 3./(1+exp(-10*(x(1,:)-2)));
%f = f - x; % and ramp

%for double parab
%f(1:round(end/2)-1) = -(x(1:round(end/2)-1)-0.5).^2;
%f(round(end/2):end) = f(round(end/2):end) + 2*(x(round(end/2):end)-3).^2;
%f = smooth(f,3)';
%f with variable frequency
%f = sin(x.^2);


%add noise
snr = 0.10;
y = f + snr*randn(size(f));

%f(x > max(x)/2 + 0.5*r) = 3*snr*randn(size(f(x > max(x)/2 + 0.5*r)))...
                           % + f(x > max(x)/2 + 0.5*r);

%plot
h = surf(0:4/(n-1):4,0:4/(n-1):4,vec2mat(f,n));
alpha(0.3);
hold on
plot3(x(1,:),x(2,:),y,'x')

%splits 100 pts into m sections and does process, then compare to doing
%whole thing
pointsPerCluster = round(2*sqrt(length(x)));
pointsPerCluster = 20;
numCluster = 4^2;

%creates cluster centers
clusterCenter = b1:(b2-b1)/(round(numCluster^(1/size(x,1)))-1):b2;
if size(x,1) == 2
    clusterCenter = kron([ones(1,length(clusterCenter)); zeros(1,length(clusterCenter))],clusterCenter) + ...
        kron(clusterCenter,flipud([ones(1,length(clusterCenter)); zeros(1,length(clusterCenter))]));
elseif size(x,1) == 3
    clusterCenter = kron([ones(1,length(clusterCenter)); zeros(1,length(clusterCenter))],clusterCenter) + ...
        kron(clusterCenter,flipud([ones(1,length(clusterCenter)); zeros(1,length(clusterCenter))]));
    cc = b1:(b2-b1)/(round(numCluster^(1/size(x,1)))-1):b2;
    
    clusterCenter = kron(ones(1,numCluster^(1/size(x,1))),[clusterCenter; zeros(1,length(clusterCenter))]) + ...
        kron(cc,flipud([ones(1,length(clusterCenter)); zeros(2,length(clusterCenter))]));
end
%clusterDist = norm(clusterCenter(:,1) - clusterCenter(:,2));
clusterDist = 1.5;
tic
for partition = 1:numCluster
    fprintf('Partition %d \n', partition);
    clear xtrunc ytrunc Syu Suu
    

    xmean = clusterCenter(:,partition);
    xtrunc = [];
    ytrunc = [];
    for j = 1:length(y)
        if norm(x(:,j) - xmean) < clusterDist
            xtrunc = [xtrunc, x(:,j)];
            ytrunc = [ytrunc, y(:,j)];
        end
    end
    
    if partition >1
        [param] = minLG(xtrunc,ytrunc,[paramHist(partition-1,:)]*0.9,1000,1);
    else
        [param] = minLG(xtrunc,ytrunc,[1 0.1],1000,1);
    end
    
    A = param(1);
    s = param(2);
    for i = 1:length(xtrunc)
        for j = 1:length(xtrunc)
           Suu(i,j) = kernel(x(:,i),x(:,j),[param(1) param(2)]);
        end
    end
    for i = 1:length(xtrunc)
        for j = 1:length(xtrunc)
           Syu(i,j) = kernel(x(:,i),x(:,j),[param(1) 0]); 
        end
    end
    yhat = Syu*Suu^-1*ytrunc';
    sigmax = Syu - Syu*Suu^-1 * Syu';
    sxx = diag(sigmax);
    %errorbar(xtrunc,yhat,0.5*sxx.^0.5,'g')
    paramHist(partition,:) = param;
    xmeanHist(:,partition) = xmean;
end
toc
%fit function with gp
tic
[paramStat] = minLG(x, y,[1 0.2],1000,1);
toc
%%
clear Suu Syu Syy
for i = 1:length(x)
    for j = 1:length(x)
       Suu(i,j) = kernel(x(i),x(j),[paramStat(1) paramStat(2)]);
    end
end
for i = 1:length(x)
    for j = 1:length(x)
       Syu(i,j) = kernel(x(i),x(j),[paramStat(1) 0]); 
    end
end
yhat = Syu*Suu^-1*y';
sigmax = Suu - Syu*Suu^-1 * Syu';
sxx = diag(sigmax);
%errorbar(x,yhat,0.5*sxx.^0.5,'r')
paramHist = abs(paramHist)
param = abs(paramStat)
A = paramStat(1);
figure
hold on
plot3(xmeanHist(1,:),xmeanHist(2,:),paramHist(:,1),'g x')
plot3(xmeanHist(1,:),xmeanHist(2,:),ones(1,numCluster)*A,'r x')



%fit gp to length scale func
[paramGPL] = minLG(xmeanHist,paramHist(:,1)',[1 1 0.2],1000,1);
[paramGPSNR] = minLG(xmeanHist,paramHist(:,2)',[1 1 0.2],1000,1);
clear Syu Suu Syy

%create more points to plot
gpLx = x;
for i = 1:length(xmeanHist)
    for j = 1:length(xmeanHist)
       Suu(i,j) = kernel(xmeanHist(:,i),xmeanHist(:,j),[paramGPL(1) paramGPL(2) paramGPL(3)]);
    end
end
for i = 1:length(gpLx)
    for j = 1:length(xmeanHist)
       Syu(i,j) = kernel(gpLx(:,i),xmeanHist(:,j),[paramGPL(1) paramGPL(2) 0]); 
    end
end
for i = 1:length(gpLx)
    for j = 1:length(gpLx)
       Syy(i,j) = kernel(gpLx(:,i),gpLx(:,j),[paramGPL(1) paramGPL(2) paramGPL(3)]); 
    end
end

gpLy = Syu*Suu^-1*paramHist(:,1);
sigmax = Syy - Syu*Suu^-1 * Syu';
sxx = diag(sigmax);
surf(range,range,vec2mat(gpLy,n));
alpha(0.3);
%errorbar(gpLx,gpLy,0.5*sxx.^0.5,'g')

%plot inverse

%figure 
%hold on
%plot(xmeanHist,1./abs(paramHist(:,1)),'g x')
%plot(xmeanHist,ones(1,numCluster-1)/A,'r')

sxx = diag(sigmax);
%errorbar(gpLx,1./abs(gpLy),sxx.^0.5,'g')


% fits data with length scale dependent on x

figure
%plot
h = surf(0:4/(n-1):4,0:4/(n-1):4,vec2mat(f,n));
alpha(0.3);
hold on
plot3(x(1,:),x(2,:),y,'x')
clear Syu Suu Syy

%plot stationary guess
for i = 1:length(x)
    for j = 1:length(x)
       Suu(i,j) = kernel(x(:,i),x(:,j),[paramStat(1) paramStat(2)]);
    end
end
for i = 1:length(x)
    for j = 1:length(x)
       Syu(i,j) = kernel(x(:,i),x(:,j),[paramStat(1) 0]); 
    end
end
for i = 1:length(x)
    for j = 1:length(x)
       Syy(i,j) = kernel(x(:,i),x(:,j),[paramStat(1) paramStat(2)]); 
    end
end
yhat = Syu*Suu^-1*y';
sigmax = Syy - Syu*Suu^-1 * Syu';
sxx = diag(sigmax);
%plot
surf(range,range,vec2mat(yhat,n));
set(h,'FaceColor',[1 0 1])
alpha(0.3);

%RMSE error
RMSE_STAT = sqrt(sum((yhat'-f).^2));
%plot nonstationary guess

%Gp guess of length parameter
clear Syu Suu Syy
for i = 1:length(xmeanHist)
    for j = 1:length(xmeanHist)
       Suu(i,j) = kernel(xmeanHist(:,i),xmeanHist(:,j),[paramGPL(1) paramGPL(2) paramGPL(3)]);
    end
end
for i = 1:length(x)
    for j = 1:length(xmeanHist)
       Syu(i,j) = kernel(x(:,i),xmeanHist(j),[paramGPL(1) paramGPL(2) 0]); 
    end
end
s = Syu*Suu^-1*paramHist(:,1);
%s = s+mean(s); %bias tends to help
for i = 1:length(s)
    if s(i) < 0.01
        s(i) = 0.01;
    end
end



%Gp guess of SNR
clear Syu Suu Syy
for i = 1:length(xmeanHist)
    for j = 1:length(xmeanHist)
       Suu(i,j) = kernel(xmeanHist(:,i),xmeanHist(:,j),[paramGPSNR(1) paramGPSNR(2) paramGPSNR(3)]);
    end
end
for i = 1:length(x)
    for j = 1:length(xmeanHist)
       Syu(i,j) = kernel(x(:,i),xmeanHist(:,j),[paramGPSNR(1) paramGPSNR(2) 0]); 
    end
end
sSNR = Syu*Suu^-1*paramHist(:,2);


%fitting data
%s values blow
clear Syu Suu Syy
for i = 1:length(x)
    for j = 1:length(x)
       Suu(i,j) = kernelNonStat(x(:,i),x(:,j),x,s(i),s(j),mean(paramHist(:,2)) );
    end
end
for i = 1:length(x)
    for j = 1:length(x)
       Syu(i,j) = kernelNonStat(x(:,i),x(:,j),x,s(i),s(j),0 );
       if isreal(Syu(i,j)) == 0
           return;
       end
    end
end
Syy = Suu;
yhat = Syu*Suu^-1*y';
sigmax = Syy - Syu*Suu^-1 * Syu';
sxx = diag(sigmax);
h = surf(range,range,vec2mat(yhat,n));
set(h,'FaceColor',[0 1 0])
alpha(0.3);

%MSE error
RMSE_NONSTAT = sqrt(sum((yhat'-f).^2))
RMSE_STAT



%plotting more data
%{
xPlot = [-1:0.05:-0.05 , x, 4.05:0.05:5];
%Gp guess of length parameter
clear Syu Suu Syy
for i = 1:length(xmeanHist)
    for j = 1:length(xmeanHist)
       Suu(i,j) = kernel(xmeanHist(i),xmeanHist(j),[paramGPL(1) paramGPL(2) paramGPL(3)]);
    end
end
for i = 1:length(xPlot)
    for j = 1:length(xmeanHist)
       Syu(i,j) = kernel(xPlot(i),xmeanHist(j),[paramGPL(1) paramGPL(2) 0]); 
    end
end
sPlot = Syu*Suu^-1*paramHist(:,1);
sPlot = sPlot+1/2*mean(s);


clear Suu Syu Syy
for i = 1:length(x)
    for j = 1:length(x)
       Suu(i,j) = kernelNonStat(x(i),x(j),xPlot,sPlot,mean(paramHist(:,2)) );
    end
end
for i = 1:length(xPlot)
    for j = 1:length(x)
       Syu(i,j) = kernelNonStat(xPlot(i),x(j),xPlot,sPlot,0 );
       if isreal(Syu(i,j)) == 0
           return;
       end
    end
end

yhat = Syu*Suu^-1*y';
plot(xPlot,yhat,'g')
%}

