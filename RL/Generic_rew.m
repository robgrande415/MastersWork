function [rew,breaker] = Generic_rew( s_old,action,params,mdp_num )
if nargin < 4
    mdp_num = 1;
end
%GENERIC_REW Summary of this function goes here
%   Detailed explanation goes here
    rew = 0; breaker= false;
    if(strcmp(params.env,'bandit')==1)
        [rew,breaker] = twoArmedBandit_rew(s_old,action,params);
    elseif(strcmp(params.env,'grid')==1)
        [rew,breaker] = gridworld_rewNeg(s_old, action,params);
    elseif(strcmp(params.env,'cartpole')==1)
        [rew,breaker] = cartpole_rew(s_old,action,params); 
    elseif(strcmp(params.env,'acrobot')==1)
        [rew,breaker] = acrobot_rew(s_old,action,params);   
    elseif(strcmp(params.env,'puddle')==1)
        [rew,breaker] = ClassicPuddleWorldReward(s_old,action,params,mdp_num); 
    elseif(strcmp(params.env,'glider')==1)
        [rew,breaker] = glider_rew(s_old,action,params);
    elseif(strcmp(params.env,'747')==1)
        [rew,breaker] = airplane_747_rew(s_old,action,params);
    elseif(strcmp(params.env,'f16')==1)
        [rew,breaker] = airplane_f16_rew(s_old,action,params);
    elseif(strcmp(params.env,'LTI')==1)
        [rew,breaker] = LTI_rew(s_old,action,params);
    elseif(strcmp(params.env,'wind')==1)
        [rew,breaker] = wind_rew(s_old,action,params,mdp_num);
    end
end
