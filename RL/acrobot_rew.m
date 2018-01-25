function [ rew, breaker ] = acrobot_rew( old_s, action, params )

theta1 = old_s(1);
theta2 = old_s(2);
y_acrobot(1) = 0;
theta1 = theta1-pi/2;
theta2 = theta2+theta1;
y_acrobot(2) = y_acrobot(1) - cos(theta1);
y_acrobot(3) = y_acrobot(2) - cos(theta2+theta1);    
height = sin(theta1) + sin(theta2);



%goal
goal = y_acrobot(1) + 1.0 ;
%RCG: edit to make harder
%goal = y_acrobot(1) + 1.8 ;
rew =  -1; %y_acrobot(3);
%edit by Rob: since we are integrating by 4 time steps, reward should be -4
breaker = false;
rew = (-2+height)/4;
if( height >= goal) 
	rew = 0; %10*y_acrobot(3);      %tjw: was 100,    
    breaker = true;
end


end

