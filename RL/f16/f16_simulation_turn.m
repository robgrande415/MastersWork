%AERSP 518 Dynamics and Control of Aerosapce Vehicles
%Script for executing 6-DOF simulation of F-16
clear all;
format compact;
%Output variables will be returned in the global variable OUTPUT
global OUTPUT;
global XCG;
d2r=pi/180.;

%%%%%%%%%%%%%%%%%%%
% Set CG position %
%%%%%%%%%%%%%%%%%%%
XCG=0.35;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Set final time of simulation here%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tfinal=  72.;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Set simulation time step here%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dt=  0.02;

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
alt=20000.;
%Set initial North-South and East-West position
p_north=0.;
p_east=0.;
%Turn Rate in deg/sec
turnrate = 5.;
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

%Trimmed state and controls are stored in x0 and u0
[A,B,C,D] = linearize('f16',x0,u0);
%Linear model stored in A and B, C and D

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Enter the name of the function that desribes state equations here%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fdyn='f16';
finp='hold_trim';

ns=length(x0);
x=x0;
ni=length(u0);
u=u0;
[xdot,y]=feval(fdyn,0.,x0,u0);
no=length(y);

time=[0:dt:tfinal];
nt=length(time);
xrec=zeros(ns,nt);
urec=zeros(ni,nt);
yrec=zeros(no,nt);

tdisp=5.;
tdinc=5.;

disp('Performing time history simulation');
for i=1:nt;
    t=time(i);
    xrec(:,i)=x;
    u=feval(finp,t,u0);
    urec(:,i)=u;
    if (no >0)
     [xdot,y]=feval(fdyn,t,x,u);
     yrec(:,i)=y;
    end;
 
    x=rk4(fdyn,finp,t,dt,x,u0);
    if (t>=tdisp)
        disp(['Time = ',num2str(t)]);
        tdisp=tdisp+tdinc;
    end;
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Put plot commands here (or just type them in the workspace after you run the script)%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%3-D Plot of Aircraft Trajectory
plot3(xrec(11,:),xrec(10,:),xrec(12,:));
xlabel('East-West Position (ft)');
ylabel('North-South Position (ft)');
zlabel('Altitude (ft)');
axis equal;