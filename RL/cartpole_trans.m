function s_new = cartpole_trans(s_old, action, params)
% paramVec = [M, m, I, l, g]


paramVec = params.paramVec;
M = paramVec(1);
m = paramVec(2);
I = paramVec(3);
l = paramVec(4);
g =paramVec(5);
b = paramVec(6);
dt = 0.025;

%action
F = 0;
if(action == 1)
    F = 0;
elseif(action ==2)
    F = -1;
elseif(action ==3)
    F = 1;
elseif(action ==4)
    F = -5;
elseif(action ==5)
    F = 5;
elseif(action ==6)
    F = -10;
elseif(action ==7)
    F = 10;
end
F = F*5;


%constraints
x_lim(1,:) = [-1 1]*10;
x_lim(2,:) = [-1 1]*10;
x_lim(3,:) = [-1 1]*4*pi;
x_lim(4,:) = [-1 1]*4*pi;


x = s_old(1);
xdot = s_old(2);
th = s_old(3);
thdot = s_old(4);
dydt(1,1) = xdot;
dydt(3,1) = thdot;

dydt(4,1) = m*l*(cos(th) * ((F- b*xdot) - m*l*(thdot^2)*sin(th)) + g*(m+M)*sin(th))/...
                (  (M+m)*(I+m*l^2) - m^2*l^2*cos(th)^2) -b*thdot;
dydt(2,1) = ((F- b*xdot) + m*l*dydt(4,1)*cos(th) - m*l*thdot^2*sin(th) )/(M+m);

s_new = s_old + dydt*dt;

%add noise
s_new(2) = s_new(2) + randn*0.01;
s_new(4) = s_new(4) + randn*0.01;

   

%edit to make position not matter, i.e. 3d
%s_new(1) = 0;
%wraps around so top is okay
%s_new(3) = mod(s_new(3)+pi+0.01,2*pi)-pi-0.01;
%wraps around so bottom is okay
s_new(3) = mod(s_new(3),2*pi);

%limits
%constraints
x_lim(1,:) = [-1 1]*10;
x_lim(2,:) = [-1 1]*10;
x_lim(3,:) = [-1 1]*4*pi;
x_lim(4,:) = [-1 1]*8*pi;
if abs(s_new(1)) > 6
    s_new(1) = 6*sign(s_new(1));
    s_new(2) = 0;
    %s_new(2) = 10*sign(s_new(2));
end
if abs(s_new(4)) > 8*pi
    s_new(4) = 8*pi*sign(s_new(4));
end

end