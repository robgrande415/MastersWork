function [nextState] = ClassicPuddleWorldStep(curState,action,params)
%PUDDLEWORLDSTEP Summary of this function goes here
%   Detailed explanation goes here

limits = [0,1;0,1];
step = 0.1;
noiseMax = 0.01;  %0.01

sign = 1;
if(randi(2) <2)
    sign = -1;
end

if(action == 1) %up
    nextState = curState + [0;step + sign * rand(1) * noiseMax]; %+ [0,(step + rand() * noiseMax)];
elseif(action ==2) %down
    nextState = curState - [0;step + sign * rand(1) * noiseMax]; %- [0,(step + rand() * noiseMax)];
elseif(action ==3) %right
    nextState = curState + [step + sign * rand(1) * noiseMax;0];%+  [(step + rand() * noiseMax), 0];
elseif(action ==4) %left
    nextState = curState - [step + sign * rand(1) * noiseMax;0];%- [(step + rand() * noiseMax), 0];
end

if(nextState(1) < limits(1,1))
    nextState(1) = limits(1,1);
end
if(nextState(1) > limits(1,2))
    nextState(1) = limits(1,2);
end
if(nextState(2) < limits(2,1))
    nextState(2) = limits(2,1);
end
if(nextState(2) > limits(2,2))
    nextState(2) = limits(2,2);
end
    

end

