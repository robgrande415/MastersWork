function [val,val_var] = Q_value(theta,s,a,params,gpr)

if params.approximation_on==0
    phi_sa = Q_tabular_feature(s,a,params);
    val = theta'*phi_sa;
    val_var = 0;
elseif params.approximation_on==1 || params.approximation_on==2 || params.approximation_on==4 || params.approximation_on==6
    phi_sa= Q_RBF(s,a,params);
    val = theta'*phi_sa;
    val_var = 0;
%elseif params.approximation_on==3 || params.approximation_on== 5
%    x=[s;a];
%    [mean_post var_post] = gpr.predict(x,params);      
%     val = mean_post;
%     val_var = 0;
elseif params.approximation_on==3 || params.approximation_on== 5|| params.approximation_on== 7
    x=[s;a];
    [mean_post var_post] = gpr.predict(x,params);      
     val = mean_post;
     val_var = var_post;
end

end