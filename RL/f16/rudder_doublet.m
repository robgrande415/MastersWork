function [u]=rudder_doublet(t,u0);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Define the input as a function of time here %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    u=u0;
    if (t<1)
        u=u0;
    elseif (t<1.5)
       u(4)=u0(4)+5;
    elseif (t<2)
       u(4)=u0(4)-5;
    else
       u=u0;
    end
    
    return