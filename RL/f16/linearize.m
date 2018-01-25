function [A,B,C,D]=linearize(acfunction,x0,u0);

%This module is used to linearize the governing equations.
%Used in trim algorithm and to derive a linear model for control design.

global DELXLIN DELCLIN;

ns=length(x0);
nc=length(u0);
[xdot0,y0]=feval(acfunction,0.,x0,u0);
nsd=length(xdot0);
no=length(y0);

A=zeros(nsd,ns);
B=zeros(nsd,nc);
C=zeros(no,ns);
D=zeros(no,nc);

for k=1:ns;
    x_p=x0;
    x_p(k)=x_p(k)+DELXLIN(k);
    [xdot_p1,y_p1]=feval(acfunction,0.,x_p,u0);
    x_p(k)=x_p(k)-2*DELXLIN(k);
    [xdot_p2,y_p2]=feval(acfunction,0.,x_p,u0);
    A(:,k)=(xdot_p1-xdot_p2)/(2*DELXLIN(k));
    C(:,k)=(y_p1-y_p2)/(2*DELXLIN(k));
end;

for k=1:nc;
    u_p=u0; 
    u_p(k)=u_p(k)+DELCLIN(k);
    [xdot_p1,y_p1]=feval(acfunction,0.,x0,u_p);
    u_p(k)=u_p(k)-2*DELCLIN(k);
    [xdot_p2,y_p2]=feval(acfunction,0.,x0,u_p);
    B(:,k)=(xdot_p1-xdot_p2)/(2*DELCLIN(k));
    D(:,k)=(y_p1-y_p2)/(2*DELCLIN(k));
end;

return;