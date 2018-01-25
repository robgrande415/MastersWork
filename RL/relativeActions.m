function [ da ] = relativeActions( actions, tol )
%RELATIVEACTIONS Summary of this function goes here
%   Detailed explanation goes here

    da = zeros(size(actions));
    for t=1:size(actions,1)
        if(t==1)
            lastAction = zeros(1,size(actions,2));
        else
            lastAction = actions(t-1,:);
        end
        da(t,:) = -1 * (actions(t,:) <= lastAction - tol);
        da(t,:) = da(t,:) + (actions(t,:) >= lastAction + tol);
    end

end

