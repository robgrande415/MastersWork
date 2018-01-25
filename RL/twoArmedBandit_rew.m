function [rew,breaker] = twoArmedBandit_rew(s_old,action,params)

if(action ==1)
    rew = -1;
else
    rew = -1 * params.gamma;
end

breaker = false;

end
