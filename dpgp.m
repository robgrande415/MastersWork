close all
clear;
for i = 1:5
    a(i) = 1/i*randn;
end
n = 50;
x = 0:4/(n-1):4;
f = 0*x;
for i = 1:length(a)
   f = f + a(i)*cos(i*x); 
end

%for step
f = heaviside(x-1) - heaviside(x-3);
%f = smooth(f)';
r = randn;
%f(x > max(x)/2 + 0.5*r) = 5 + f(x > max(x)/2 + 0.5*r);
plot(x,f)
hold on
%add noise
snr = 0.05;
y = f + snr*randn(size(f));
%f(x > max(x)/2 + 0.5*r) = 3*snr*randn(size(f(x > max(x)/2 + 0.5*r)))...
                           % + f(x > max(x)/2 + 0.5*r);
plot(x,y,'x')

%assigns each cluster randomly, starting with 2 clusters
clust_id = sign(randn(size(x)));
clust_id = sign(x-2);
num_cluster = 2;

clear cluster;
cluster(1).x = x(clust_id > 0);
cluster(2).x = x(clust_id <= 0);
cluster(1).y = y(clust_id > 0);
cluster(2).y = y(clust_id <= 0);
cluster(1).param = [1 1 1];
cluster(2).param = [1 1 1];
plot(cluster(2).x,cluster(2).y,'rx')



%trains gp on clusters
for i = 1:length(cluster)
    cluster(i).param = minLG(cluster(i).x,cluster(i).y,cluster(i).param,1000,0);
end

%%

alpha = 0.2; %dp parameter
%reassigns each data point
for iter = 1:5
    i=1;
    while( i <= length(cluster))
        j = 1;
        while( j <= length(cluster(i).x))
        %while( j <= 3)
            %pop from cluster i
            pt_x = cluster(i).x(1);
            pt_y = cluster(i).y(1);
            cluster(i).x(1) = [];
            cluster(i).y(1) = [];

            pr = zeros(1,length(cluster)+1);

            k = 1;
            while( k <= length(cluster))

                %if empty end
                if length(cluster(k).x) == 0
                    cluster(k) = [];
                    k = k-1;
                    continue;
                end
                param = cluster(k).param;
                Suu = zeros(length(cluster(k).x));
                for ii = 1:length(cluster(k).x)
                    for jj = 1:length(cluster(k).x)
                       Suu(ii,jj) = kernel(cluster(k).x(ii),cluster(k).x(jj),[param(1) param(2)], param(3));
                    end
                end

                if rank(Suu) < size(Suu,1)
                    return;
                end
                iSuu = Suu^-1;
                clear Syu
                
                for jj = 1:length(cluster(k).x)
                   Syu(1,jj) = kernel(pt_x,cluster(k).x(jj),[param(1) param(2)], 0); 
                end

                mu = Syu*iSuu*cluster(k).y';
                sigma2 = kernel(pt_x,pt_x,[param(1) param(2)], param(3)) - Syu*iSuu*Syu';

                %likelihood
                if sigma2 > 0
                    pr(k) = 1/(2*pi*sqrt(sigma2)) * exp( -(pt_y - mu)^2/(2 * sigma2));
                else
                    k;
                    sigma2;
                end
                %times model probability
                pr(k) = pr(k) * (length(cluster(k).x) / (length(x) - 1 + alpha));
                if (isreal(pr(k)) == false)
                    return;
                end
                k = k+1;
            end
            pr(end) = alpha / (length(x) - 1 + alpha); %probability of new process

            [asdf cluster_id] = max(pr);
            cluster_id;
            if (cluster_id > length(cluster))
                cluster(cluster_id).x = pt_x;
                cluster(cluster_id).y = pt_y;
                cluster(cluster_id).param = param;
            else
                temp = [cluster(cluster_id).x, pt_x];
                cluster(cluster_id).x = temp;
                temp = [cluster(cluster_id).y, pt_y];
                cluster(cluster_id).y = temp;
            end

            j = j+1;
        end
        i=i+1;
        %deletes empty clusters
        ct=1;
        while( ct <= length(cluster))
            param = cluster(ct).param;
            if numel(param) ==0
                cluster(ct) = [];
                continue;
            end
            ct = ct+1;
        end
    end

    %deletes empty clusters
    %trains gp on clusters
    ct=1;
    while( ct <= length(cluster))
        param = cluster(ct).param;
        if numel(param) ==0
            cluster(ct) = [];
            continue;
        end
        cluster(ct).param = minLG(cluster(ct).x,cluster(ct).y,cluster(ct).param,1000,0);
        ct = ct+1;
    end

end
figure
plot(x,f) 
hold on
plot(cluster(1).x,cluster(1).y,'x')
hold on
if length(cluster) >1
plot(cluster(2).x,cluster(2).y,'rx')
end
hold on
if length(cluster) >2
plot(cluster(3).x,cluster(3).y,'gx')
end
hold on
if length(cluster) >3
plot(cluster(4).x,cluster(4).y,'kx')
end