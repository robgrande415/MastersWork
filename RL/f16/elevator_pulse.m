function [u]=elevator_pulse(t,u0);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Define the input as a function of time here %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    u=u0;
    if (t<1)
        u=u0;
    elseif (t<=2)
       u(3)=u0(3)-1;
    else
       u=u0;
    end
    
    return