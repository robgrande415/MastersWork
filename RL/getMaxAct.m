function [maxact]=getMaxAct(q, s)
    maxact = 1;    
    for a=2:size(q,2)
        if(q(s,a) > q(s,maxact))
            maxact = a;
        end
    end
end

