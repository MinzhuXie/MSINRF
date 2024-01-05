clear all
clc
param=importdata([".\parameter.xlsx"]);
param=param;
% A=[0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95];
X=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1];
Y=[param(1,:);param(2,:);param(3,:);param(4,:);param(5,:);param(6,:);param(7,:);param(8,:);param(9,:);param(10,:)];
h=bar3(X,Y);
set(gca,'xticklabel',0.1:0.1:1);
% set(gca,'yticklabel',0.1:0.1:1);
set(gca,'zticklabel',0.90:0.05:1);
% zlim([0.90,1]);
for n=1:numel(h)
    cdata=get(h(n),'zdata');
    set(h(n),'cdata',cdata,'facecolor','interp')
end
xlabel('¦Ác');ylabel('¦Ád');zlabel('AUC');
% title('parameter experiment');

colorbar
lim = caxis
caxis([0.90 1])





