function out = DkernelDSNR(x1,x2,param, snr)
out = 0;
A = param(1);
s = param(2:end);

%if the exponent is too small, 0 returned
if (sum((x1-x2).^2)/(2*s) < 6)
    if length(param) == 2
       out = exp(-sum((x1-x2).^2)/(2*s)); 

    elseif length(param) == 3
       out = exp(-sum((x1-x2).^2)/(2*s(1))) ...
            + exp(-sum((x1-x2).^2)/(2*s(2))); 
    end

    if x1==x2
        out = out + 2*snr;
    end
end
     
        
        


end