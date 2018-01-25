function [Q_val_opt,action] = Q_greedy_act_var(theta,s,params,gpr)

N_act = params.N_act;


act_val = zeros(1,N_act);
act_val_var = zeros(1,N_act);
for l=1:N_act
    
    a = l;
    
    [act_val(l),act_val_var(l)] =  Q_value(theta,s,a,params,gpr);
    
end
if sum(act_val+2*act_val_var==max(act_val+2*act_val_var))>1
    pos = find(act_val+2*act_val_var==max(act_val+2*act_val_var));
    action = pos(randi(length(pos),1));
    Q_val_opt = act_val(action)+2*act_val_var(action);
else
    [Q_val_opt,action] = max(act_val+2*act_val_var);
end
end