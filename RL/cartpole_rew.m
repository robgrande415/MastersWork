function [ rew, breaker ] = cartpole_rew( s_old, action, params )

%constraints
x_lim(1,:) = [-1 1]*2;
x_lim(2,:) = [-1 1]*10;
x_lim(3,:) = [-1 1]*8*pi;
x_lim(4,:) = [-1 1]*8*pi;


x = s_old(1);
xdot = s_old(2);
th = s_old(3);
thdot = s_old(4);


%goal
goal = 0.9;
rew =  -1+cos(th); %continuous
%rew = -1;
%discrete
%if abs(th) > pi/9
%    rew = -1;
%end
%if abs(x) > 1
%    rew =  rew - 1/64*x^2; %y_acrobot(3);
%end
%rew = -1+exp(-1/2*th^2 - 1/16*x^2);
breaker = false;


%impose constraints
%balance
%if cos(th) <= -0.9
    %rew =rew-10;
%    breaker = true;
%    return;
%end

if sum(s_old < x_lim(:,1)) >=1
    %rew = rew-2;
    %return;
elseif sum(s_old > x_lim(:,2)) >=1
    %rew = rew-2;
    %return;
end

%swing up
if( cos(th) >= goal) 
	rew = 0; %10*y_acrobot(3);      %tjw: was 100,    
    breaker = true; %balance or just swing up?
    return;
end

end

