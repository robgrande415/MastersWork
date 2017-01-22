function out = DkernelDS(x1,x2,param, snr)
out = 0;
A = param(1);
s = param(2:end);

%if the exponent is too small, 0 returned
if (sum((x1-x2).^2)/(2*s) < 6)
    if length(param) == 2
       out = A*exp(-sum((x1-x2).^2)/(2*s)) * (-sum((x1-x2).^2)/(2*s)) * ...
                                                (sum((x1-x2).^2)/(2*s*s)) ; 

    elseif length(param) == 3
       out = A*exp(-sum((x1-x2).^2)/(2*s(1))) ...
            + A*exp(-sum((x1-x2).^2)/(2*s(2))); 
    end

    if x1==x2
        out = out + snr*snr;
    end
end
     
        
        


end