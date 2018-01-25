function [ rew, breaker ] = airplane_f16_rew( s_old, action, params )
 
global f16_actions
u = f16_actions(:,action);
breaker = 0;
rew = -abs(s_old(5)) - abs(s_old(4)+s_old(3))-abs(sum(u));

%feature stuff
s_old(5) = s_old(5)*100;
s_old(2) = s_old(2)/10;
s_old(3) = s_old(3)/10;
 %lqr reward
 P = diag([1, 0, 2, 1, 1/100, 0]);
 Q = eye(2)/100;
 
 %rew = -s_old' * P * s_old - u' * Q * u;
 
 %cap rew
 if rew < -10
     rew = -10;
 end
 
 %{
 if abs(s_old(5)) > 50
     rew = -1;
 end
  if abs(s_old(5)) > 500
     rew = -10;
 end
 %}
 %limits
 %{
 if abs(s_old(5)) > 1000
     breaker = 1;
     rew = -10;
 end
 %}
 end