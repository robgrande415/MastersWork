function [u]=elevator_step(t,u0);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Define the input as a function of time here %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    u=u0;
    if (t<1)
        u=u0;
    else
       u(3)=u0(3)-1;
    end
    
    return