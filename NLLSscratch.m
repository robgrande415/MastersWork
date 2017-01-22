clear;
%close all;

res = 0.1;
t= 0:res:2;
t = t';
w0 = 1;
y = (1+0.5*randn)*sin((1+0.5*randn)*w0*t - 2*randn);
R = 0.005;
y = y + sqrt(R)*randn(size(y));
%%
subplot(1,2,1), h=stem(t,y); title('Data');
param = [0; 1; 1; 2];
a=param(1); b=param(2); w=param(3); t0=param(4);
c=[a b w t0]'; %initial guess
n=size(t,1); iter=0;dcnorm=1.;
error = var(y);
counter = 0;
while error > var(y)/2 & counter < 1000
    while dcnorm>1E-3 & iter<100
        u=w*(t-t0); f=a+b*sin(u)-y;
        Ji1=ones(n,1); Ji2=sin(u);
        Ji3=b*(t-t0).*cos(u); Ji4=-w*b*cos(u);
        J=[Ji1 Ji2 Ji3 Ji4]; % Jacobian
        dc=-J\f; c=c+dc+randn(size(c))*(0.1*norm(c)/(2*iter+1)); % Gauss-Newton
        dcnorm=norm(dc)/norm(c); iter=iter+1;
        a=c(1); b=c(2); w=c(3); t0=c(4);
        D=[iter a b w t0 norm(f) norm(dc) ];
        Ft=a+b*sin(w*(t-t0));
        error = sum((Ft-y).^2)/length(y);
    end
    
    if error > var(y)/2
        %perturbs y
        %y = y + 0.001*randn(size(y));
        a=param(1); b=param(2); w=param(3)+0.1*randn; t0=param(4);
        c=[a b w t0]'; %initial guess
        n=size(t,1); iter=0;dcnorm=1.;
    end
    
    counter = counter +1;
end

D;
error
var(y)
tt = min(t):0.01:max(t);
Ft=a+b*sin(w*(tt-t0));
clf;
plot(t,y,'o',tt,Ft)
    