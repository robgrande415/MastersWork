function [s_new,params] = Generic_trans( s_old,action,params,mdp_num)
%GENERIC_TRANS Summary of this function goes here
%   Detailed explanation goes here
    s_new = s_old;
    if(strcmp(params.env, 'bandit') == 1)
        s_new = 1;
    elseif(strcmp(params.env, 'grid') == 1)
        s_new = gridworld_trans(s_old,action,params);
    elseif(strcmp(params.env, 'acrobot') == 1)
        s_new = acrobot_trans(s_old,action,params)';
    elseif(strcmp(params.env, 'cartpole') == 1)
        s_new = cartpole_trans(s_old,action,params);
    elseif(strcmp(params.env, 'puddle') == 1)
        s_new = ClassicPuddleWorldStep(s_old,action,params);
    elseif(strcmp(params.env, 'glider') == 1)
        [s_new,params] = glider_trans(s_old,action,params);
    elseif(strcmp(params.env, '747') == 1)
        s_new = airplane_747_trans(s_old,action,params);
    elseif(strcmp(params.env, 'f16') == 1)
        s_new = airplane_f16_trans(s_old,action,params);
    elseif(strcmp(params.env, 'LTI') == 1)
        s_new = LTI_trans(s_old,action,params);
    elseif(strcmp(params.env, 'wind') == 1)
        s_new = wind_trans(s_old,action,params,mdp_num);
    end
end

