function [xnew]= rk4(fdyn,finp,time,dt,xcurr,u0)

%Performs 4th order Runge-Kutta Integration of the equations xdot=f(t,x,u)
% Function fdyn is dynamic model.
% Function finp determines input as a function of time.
% Variable time is current time (used to find input and for time varying dynamics).
% Time step  = dt
% Value of state vector at current time = xcurr
% Returns state vector at time t+dt

u=feval(finp,time,u0);
xd=feval(fdyn,time,xcurr,u);
k1=dt*xd;
u=feval(finp,time+0.5*dt,u0);
xd=feval(fdyn,time+0.5*dt,xcurr+0.5*k1,u);
k2=dt*xd;
xd=feval(fdyn,time+0.5*dt,xcurr+0.5*k2,u);
k3=dt*xd;
u=feval(finp,time+dt,u0);
xd=feval(fdyn,time+dt,xcurr+k3,u);
k4=dt*xd;

xnew=xcurr+k1/6+k2/3+k3/3+k4/6;

return;