function s = animate_cart(t,y,r)
close all;
figure

for i = 2:length(t)
    y(i,1) = y(i-1,1) + y(i-1,2)*0.025;
end
for i = 1:length(t)
   clf;
   plot([0 -sin(y(i,3))]+y(i,1),[0 cos(y(i,3))]);
   hold on
   h = plot(y(i,1),0,'x');
   set(h,'MarkerSize',10);
   set(h,'LineWidth',3);
   circle(-sin(y(i,3))+y(i,1),cos(y(i,3)),0.2);
   if nargin ==3
    plot([0 -sin(r(i,3))]+r(i,1),[0 cos(r(i,3))],'r');
    hold on
    circle(-sin(r(i,3))+r(i,1),cos(r(i,3)),0.2,'r');
   end
   axis([-1 1 -1 1]*7);
   pause(0.05);
end
s = 0;
end