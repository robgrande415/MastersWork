function [ ret, breaker ] = ChainRew( snew, params )
%CHAINREW Summary of this function goes here
%   Detailed explanation goes here
    breaker = false;
    if snew == 1
        ret = -1;
    elseif snew == 2
        ret = 1;
    else
        ret =  5;
end

