function [ snew ] = ChainTrans( s_old,action,params )
%CHAINTRANS Summary of this function goes here
%   Detailed explanation goes here
    if action == 1
        snew = 2;
    else
        if(rand() < 0.4)%go up
            snew = min(s_old + 1, params.N_state);
        else
            snew = 1;%go to 1
    end



end

