% GP-SARSA
clear all;
close all;

theta = 1;
wn = 0.1;
Sigma = wn;
%step one
Q = 0;
alpha = 0;
r1 = -1;
gamma = 0.9;
Rvec = r1;
r2 = -0.9;

%repeat 
max_iter = 100;
Qstack = zeros(1,max_iter);
for i = 2:max_iter
    
    Rvec = [Rvec; r1];
    H = [eye(i),zeros(i,1)];
    temp = [zeros(i,1),eye(i)*(-gamma)];
    H = H+temp;
    K = ones(i+1);
    Sigma = wn*eye(i);
    alpha = H' * (H * K * H' + Sigma)^-1 * Rvec;
    Qstack(i) = sum(alpha);
end

plot(Qstack)