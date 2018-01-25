function [Q_val_opt,action] = Q_greedy_act(theta,s,params,gpr)

N_act = params.N_act;


act_val = zeros(1,N_act);
act_val_var = zeros(1,N_act);
for x=1:N_act
    
    
    [act_val(x),act_val_var(x)] =  Q_value(theta,s,x,params,gpr);
    
end
if sum(act_val==max(act_val))>1
    pos = find(act_val==max(act_val));
    action = pos(randi(length(pos),1));
    Q_val_opt = act_val(action);
else
    [Q_val_opt,action] = max(act_val);
end
end