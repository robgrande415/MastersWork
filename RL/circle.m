function circle(x,y,r,prop)
%x and y are the coordinates of the center of the circle
%r is the radius of the circle
%0.01 is the angle step, bigger values will draw the circle faster but
%you might notice imperfections (not very smooth)
ang=0:2*pi/50:2*pi; 
xp=r*cos(ang);
yp=r*sin(ang);
if nargin < 4
    prop = 'b';
end
plot(x+xp,y+yp,prop);
end