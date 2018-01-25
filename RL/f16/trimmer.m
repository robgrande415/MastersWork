function [x0trim,u0trim,itrim]=trimmer(acfunction,x0,u0,targ_des);

%This module is used to calculate a trim (or equilibrium) condition for a dynamic model

global TOL TRIMVARS TRIMTARG;

ns=length(x0);
nc=length(u0);
x0trim=x0;
u0trim=u0;

it=0;
itmax=500;
conv=0;

disp('Trimming:')
disp('Iteration    Error');
disp('------------------');

while ((it<itmax)&(~conv));
 it=it+1;
 [xdot0,y]=feval(acfunction,0.,x0trim,u0trim);
 targvec=[xdot0;y];
 targvec=targvec(TRIMTARG);
 targ_err=targvec-targ_des;
 
 disp([num2str(it), '            ' ,num2str(norm(targ_err))]);
 
 conv=1;
 for k=1:ns;
     if (abs(targ_err(k))>TOL(k)) conv=0; end;
 end;
 
 [A,B,C,D]=linearize(acfunction,x0trim,u0trim);
 
 if (~conv);
  Jac=[A B;C D];
  Jac=Jac(TRIMTARG,TRIMVARS);
  trimvec=[x0trim;u0trim];
  trimvec(TRIMVARS)=trimvec(TRIMVARS)-0.5*pinv(Jac)*targ_err;;
  x0trim=trimvec(1:ns);
  u0trim=trimvec(ns+1:ns+nc);
 end;
 
end;
 
if (~conv);
    disp('Warning: Trim not acheived');
    itrim=0;
else;
    disp(['Successful Trim']);
    itrim=1;
end;

return;