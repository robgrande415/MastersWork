function [ rew, goal ] = ClassicPuddleWorldReward(curState,act, params,mdp_num)
%PUDDLEWORLDREWARD Summary of this function goes here
%   Detailed explanation goes here

%adapted from the RL-GLue implementation of puddle world
%http://code.google.com/p/rlglue/source/browse/trunk/env/Puddleworld/PuddleWorld.cpp?r=124

curState = curState';

numPuddles = 2;
puddleRadius = 0.1;
Puddles = zeros(2,2,2);
Puddles(1, 1 ,1) =  0.10; %puddle #1, point #1, x
Puddles(1,1, 2) =  0.75; %puddle #1, point #1, y
Puddles(1,2,1) =  0.45; %puddle #1, point #2, x
Puddles(1,2,2) =  0.75; %puddle #1, point #2, y

   
Puddles(2,1,1) = 0.45; %puddle #2, point #1, x
Puddles(2,1,2) = 0.40; %puddle #2, point #1, y
Puddles(2,2,1) = 0.45; %puddle #2, point #2, x
Puddles(2,2,2) = 0.80; %puddle #2, point #2, y

%original code
if 0
    goalLoc = [1.0, 1.0];
    thresh  = 0.15;

    rew = -1;
    goal = false;
    if(norm(curState - goalLoc) <= thresh)
        rew = 0;
        goal = true;
        return;
    else
        for p=1:numPuddles 
            d = distance(curState, p, Puddles);
            if ( d < puddleRadius)
                rew = rew - 400/10 * (puddleRadius - d);
            end
        end
    end
else
    %1st scenario
    %goal loc
    switch mdp_num
        case 1
            goalLoc = [1.0, 1.0];
        case 2
            goalLoc = [0.2, 1.0];
        case 3
            goalLoc = [1.0, 0];
        case 4
            goalLoc = [0, 0];
    end

    %puddle place
    switch mdp_num
        case 1
            puddleLoc = [1, 0.5];
        case 2
            %puddleLoc = [1, 0.5];
            puddleLoc = [0.5, 1];
        case 3
            puddleLoc = [0, 0.5];
        case 4
           puddleLoc = [0.5, 0];
    end
    goalLoc = [1,1];
    rew = -1; %step cost
    %rew = -norm(curState-goalLoc,1)/2;
    rew = rew-2*exp(-8*(norm(curState-puddleLoc))^2); %puddle
    %if mdp_num ==2
    %    rew = rew+2.5*exp(-4*(norm(curState-puddleLoc))^2);
    %end


    %second case
    %{
    switch mdp_num
        case 1
            goalLoc = [1, 1];
            goalLoc2 = [1, 1];
        case 2
            goalLoc = [1, 1];
            goalLoc2 = [0.6, 0.6];
            goalLoc2 = [1, 1];
        case 3
            goalLoc = [1, 1];
            goalLoc2 = [1, 1];
        case 4
            goalLoc = [0.5, 0.5];
    end
    %}
    %goalLoc = [1,1];
    %rew = -1;
    thresh  = 0.15;

    rew(rew>0) = 0;
    goal = false;
    if(norm(curState - goalLoc) <= thresh)
        rew = 0;
        goal = true;
        return;
    end

    %second goal
    %{
    rew(rew>0) = 0;
    goal = false;
    if(norm(curState - goalLoc2) <= thresh)
        rew = 0;
        goal = true;
        return;
    end
    %}

    %EDIT BY ROB
    %rew = rew/40;
    %fprintf('returning reward %f at %f , %f\n',rew, curState(1), curState(2));    
end

end


function d = distance(state, p, P)
  v = zeros(1,2);
  w= zeros(1,2);
  z= zeros(1,2);
  d = 0.0;
  
  for i=1:2
    v(i) = P(p ,2 , i) - P(p,1,i);
    w(i) = state(i) - P(p,1,i);
  end

  c1 = w * v';
  if ( c1 <= 0 ) 
    d = 0.0;
    for i=1:2
      d  = d +  (state(i) - P(p,1,i)) * (state(i) - P(p,1,i));
      d = sqrt(d);
    end
    return
  end
  c2 = v * v';
  if ( c2 <= c1 ) 
    d = 0.0;
    for i=1:2
      d  = d +  (state(i) - P(p,2,i)) * (state(i) - P(p,2,i));
    d = sqrt(d);
    end
    return
  end

  b = c1 / c2;
  d = 0.0;
  for i=1:2
    z(i) =  P(p,1,i) + b * v(i);
    d = d + (state(i) - z(i)) * (state(i) - z(i));
  end
  d = sqrt(d);
  return;

end

