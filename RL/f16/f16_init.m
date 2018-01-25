%AERSP 518 Dynamics and Control of Aerosapce Vehicles
%Script for initializing Simulink Version of 6-DOF simulation of F-16
% Just set trim CG and trim conditions as you would with MATLAB model
%clear all;
format compact;

global XCG;
global X0TRIM U0TRIM;
d2r=pi/180.;

%%%%%%%%%%%%%%%%%%%
% Set CG position %
%%%%%%%%%%%%%%%%%%%
XCG=0.35;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Aircraft Trim Solution: Set initial trim conditions here%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Set velocity in ft/sec
Speed=500.;
%Vertical flight path in deg
gamma=0.;
%Horizontal flight path in deg
chi=0.;
%Set initial altitude
alt=10000.;
%Set initial North-South and East-West position
p_north=0.;
p_east=0.;
%Turn Rate in deg/sec
turnrate = 0.;
%Pull-up rate in deg/sec
pullup = 0.;
%Roll rate in deg/sec
rollrate=0.;

%Initial guess of state and control vector
phi_est=turnrate*d2r*Speed/32.17;
pndot=Speed*cos(gamma*d2r)*cos(chi*d2r);
pedot=Speed*cos(gamma*d2r)*sin(chi*d2r);
hdot=Speed*sin(gamma*d2r);
x0=[Speed;0;0;phi_est;0;chi*d2r;0.;0.;0.;p_north;p_east;alt;0.];
u0=[0.2;0;0;0];
targ_des=[0;0;0;rollrate*d2r;pullup*d2r;turnrate*d2r;0;0;0;pndot;pedot;hdot;0;0];

global DELXLIN DELCLIN TOL TRIMVARS TRIMTARG;
DELXLIN=[0.1;0.001;0.001;0.001;0.001;0.001;0.001;0.001;0.001;0.1;0.1;0.1;0.1];
DELCLIN=[0.1;0.1;0.1;0.1];
TOL=1e-5*[1;d2r;d2r;d2r;d2r;d2r;d2r;d2r;d2r;1;1;1;1;1];
TRIMVARS=[1:9 13 14 15 16 17];
TRIMTARG=[1:13 15];
[x0,u0,itrim]=trimmer('f16',x0,u0,targ_des);

X0TRIM=x0;
U0TRIM=u0;

