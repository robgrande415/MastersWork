clear;
gamma = 0.9;

%GP params
params.rbf_mu = 0.01;
params.sigma = 0.01;
params.N_budget = 200;
params.tol = 0.001;
params.A = 1;

actionTol = 0.25;

load('Helix_5.mat');

% get position and velocity
r=Act_State.signals.values(:,1:3);  % position
vel=Act_State.signals.values(:,4:6);
%get e and edot
e=Trackingerror.signals.values; %Eq 3.7
e_dot=Trackingerror_dot.signals.values; %Eq 3.8

% get the total acceleration
m=1.5; %mass of this particular body
F=Fi.signals.values; % force vector in inertial frame
r_dotdot=F/m;  %  % Eq 3.9

rdotdot_fb=rdotdot_fb.signals.values; % Eq 3.6

%get desired and actual Quaternions
Qa=Attitude.signals.values(:,1:4);
Qd=Attitude.signals.values(:,5:8);  % Eq 3.14

%error Quaternion
qe=Qe.signals.values; % Eq 3.22

% control inputs
Fb=Control_Inputs.signals.values(:,1); 
M=Control_Inputs.signals.values(:,2:4);





states = [r,vel,rdotdot_fb,Qa,Qd]; %[data.posx ,  data.posy , data.posz];

actions = F(:,3);%[data.controlinputF, data.controlinputMx, data.controlinputMy, data.controlinputMz];

%dactions = discretizeActions(actions,3);
dactions = relativeActions(actions,actionTol);
uniqueActions = sort(unique(dactions));

params.N_act = size(uniqueActions,1);

gpr1 = onlineGP.onlineGP_MA(params.N_act,params.rbf_mu,params.sigma,params.N_budget,params.tol,params.A);


%train the GP
for t=1:size(states,1)-1
    reward = -1 * norm(e(t,:)); 
    [qnext,action] =  gpr1.getMax(states(t+1,:)');
    gpr1.update(states(t,:)', find(uniqueActions == dactions(t,:)), reward + gamma * qnext);
end

%executing the trained policy
%for t=1:numSteps
   %s = current State fom simulator: [r, vel, rdotdot_fb, Qa, Qd]
   %[Q_opt,a] = gpr1.getMax(s_old);
   %act = transform "a" from line above into an actual command (1 =
   %decrease, 2= same, 3 = increase)
   %execute act in simulator
%end

