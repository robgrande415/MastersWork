function s = animate_airplane(t,y,act)
close all;
figure
%747
%height = y(:,5);
%theta = y(:,4);
%f16
elev = 15/180*pi;
elev2 = 5/180*pi;

y(:,1) = y(:,1)*10;
y(:,2) = y(:,2)/100;
y(:,3) = y(:,3)/100;
y(:,4) = y(:,4)/100;
height = y(:,5)*100;
theta = y(:,3);
for i = 1:length(t)
   clf;
   %longitudal
   L = 50;
   xvec = [0-L*cos(theta(i)),0+L*cos(theta(i))];
   yvec = [height(i)-L*sin(theta(i)),height(i)+L*sin(theta(i))];
   plot(xvec,yvec,'LineWidth',4)
   hold on
   plot(xvec,yvec,'x','LineWidth',4)
   axis([-1 1 -2 2]*205)
   L2 = 60;
   if act(i) == 2
       xvec = [-L*cos(theta(i)),-L*cos(theta(i))-L2*cos(theta(i)-elev)];
       yvec = [height(i)-L*sin(theta(i)),height(i)-L*sin(theta(i))-L2*sin(theta(i)-elev)];
        plot(xvec,yvec,'g','LineWidth',2)
        hold on
        plot(xvec,yvec,'xg','LineWidth',2)
   elseif act(i) == 3
       xvec = [-L*cos(theta(i)),-L*cos(theta(i))-L2*cos(theta(i)+elev)];
       yvec = [height(i)-L*sin(theta(i)),height(i)-L*sin(theta(i))-L2*sin(theta(i)+elev)];
        plot(xvec,yvec,'c','LineWidth',2)
        hold on
        plot(xvec,yvec,'xc','LineWidth',2)
    elseif act(i) == 4
       xvec = [-L*cos(theta(i)),-L*cos(theta(i))-L2*cos(theta(i)-elev2)];
       yvec = [height(i)-L*sin(theta(i)),height(i)-L*sin(theta(i))-L2*sin(theta(i)-elev2)];
        plot(xvec,yvec,'g','LineWidth',2)
        hold on
        plot(xvec,yvec,'xg','LineWidth',2)
        
   elseif act(i) == 5
       xvec = [-L*cos(theta(i)),-L*cos(theta(i))-L2*cos(theta(i)+elev2)];
       yvec = [height(i)-L*sin(theta(i)),height(i)-L*sin(theta(i))-L2*sin(theta(i)+elev2)];
        plot(xvec,yvec,'c','LineWidth',2)
        hold on
        plot(xvec,yvec,'xc','LineWidth',2)
   else
       xvec = [-L*cos(theta(i)),-L*cos(theta(i))-L2*cos(theta(i))];
       yvec = [height(i)-L*sin(theta(i)),height(i)-L*sin(theta(i))-L2*sin(theta(i))];
       plot(xvec,yvec,'LineWidth',2)
        hold on
        plot(xvec,yvec,'x','LineWidth',2)
   end
   
   %lateral dynamics
   %{
   %body
   [x,y,z] = sph2cart(yaw(i),pitch(i),1);
   nose = [x,y,z];
   [x,y,z] = sph2cart(yaw(i),pitch(i),-1);
   tail = [x,y,z];
   vec = [tail; nose];
   plot3(vec(:,1),vec(:,2),vec(:,3));
   hold on
   plot3(vec(:,1),vec(:,2),vec(:,3),'x');
   
   %wings
   [x,y,z] = sph2cart(yaw(i)*cos(pitch(i)) + roll(i)*sin(pitch(i)) + pi/2,...
                        -yaw(i)*sin(pitch(i)) + roll(i)*cos(pitch(i)),0.5);
   lwing = [x,y,z];
   [x,y,z] = sph2cart(yaw(i)*cos(pitch(i)) + roll(i)*sin(pitch(i)) + pi/2,...
                        -yaw(i)*sin(pitch(i)) + roll(i)*cos(pitch(i)),-0.5);
   rwing = [x,y,z];
   vec = [lwing; rwing];
   plot3(vec(:,1),vec(:,2),vec(:,3),'r');
   hold on
   plot3(vec(:,1),vec(:,2),vec(:,3),'rx');
   
   axis([-1 1 -1 1 -1 1]*1.5);
   %}
   pause(1/20);
end
s = 0;
end