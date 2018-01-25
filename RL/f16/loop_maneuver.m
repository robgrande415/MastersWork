function [u]=loop_maneuver(t,u0);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Define the input as a function of time here %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    u=u0;
    if (t<2)
        u=u0;
    elseif (t<22.)
       u(3)=u0(3)-0.9;
       u(1)=1.;       
    elseif (t<32)
       u(3)=u0(3)-0.9;
    else
        u=u0;
    end
    
    return