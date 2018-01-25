function [ da ] = discretizeActions( acts, bins )
%DISCRETIZEACTIONS Summary of this function goes here
%   Detailed explanation goes here

    daTemp = zeros(size(acts));
    dim = size(acts,2);
    da = zeros(size(acts,1),1);
    for i=1:dim
        mx= max(acts(:,i));
        mn= min(acts(:,i));
        gap = (mx - mn)/bins;
        daTemp(:,i) = min(bins-1,floor((acts(:,i) - mn) / gap));
        da(:,1) = da(:,1) + daTemp(:,i) * bins^(i-1);
    end
    da = da + 1;
    
end

