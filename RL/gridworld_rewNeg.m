function [rew,breaker] = gridworld_rew(s_new, action, params)

s_goal = params.s_goal;
rew_goal = params.rew_goal;
rew_obs = params.rew_obs;
N_obstacle = params.N_obstacle;
obs_list = params.obs_list;

breaker = false; %stop if reached goal (terminate episode)

rew = params.rew_obs;

% Check for goal
if((s_new(1) == s_goal(1))...
        && (s_new(2) == s_goal(2)))
    rew = rew_goal;
    breaker = true;
    
    
end


% Check for Obstacles

for i =1:N_obstacle
    
    obs_x = obs_list(i,1);
    obs_y = obs_list(i,2);
    
    if((s_new(1) == obs_x)...
            && (s_new(2) == obs_y))
        rew = rew + rew_obs;
    end
end
