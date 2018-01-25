function [ xp ] = airplane_f16_trans( s_old, action, params)
%airplane_TRANS Summary of this function goes here
%   Detailed explanation goes here

%feature stuff
s_old(1) = s_old(1)*10;
s_old(2) = s_old(2)/100;
s_old(3) = s_old(3)/100;
s_old(4) = s_old(4)/100;
s_old(5) = s_old(5)*100;

dt = 1/20;
%dt = 1/5;
global f16_actions
global A_f_16 B_f_16
u = f16_actions(:,action);

xp = s_old;
for i = 1:1
    xp = xp + dt*(A_f_16*xp + B_f_16*u);
    xp(1) = xp(1) +dt*randn*1;
    xp(2) = xp(2) +dt*randn*0.03;
end

if abs(xp(5)) > 200
    temp = 200*sign(xp(5));
    %xp = 0*xp;
    xp(5) = temp;
    xp(2) = 0;
    xp(3) = 0;
end
%x = X0TRIM;
%A_exp = exp(A_f_16*dt);
%dxdt = A_f_16*s_old + B_f_16*u;
%d2xdt = A_f_16*dxdt;
%xp = A_exp*s_old + (A_exp-eye(6))*A_f_16\(B_f_16*u);
%noise
%v = zeros(6,1);
%v(1) = randn*1/10;
%v(2) = randn*0.1;
%dxdt = dxdt + A_f_16*v;
%x_vel = s_old(1)*cos(s_old(2));
%y_vel = s_old(1)*sin(s_old(2));
%xp = s_old + dxdt*dt + 1/2*d2xdt*dt^2;

%feature stuff
xp(1) = xp(1)/10;
xp(2) = xp(2)*100;
xp(3) = xp(3)*100;
xp(4) = xp(4)*100;
xp(5) = xp(5)/100;

end

