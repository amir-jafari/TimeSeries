function [flag] = testarx()
%TESTARX Test ARX model.

% Yong Hu, Martin Hagan, 9-15-00
% $Revision: 1.0 $ $Date: 21-Sep-2000 14:37:36 $

%Set parameters
nb = [1 1 2]; 
na = 2;
delay = [1 2 3];
b = {[1 1] [2 3] [1 -1 .25]};
a = {[-1 0.25]};
u=randn(3,2000);
e=randn(1,2000)*.5;

pmoda = newarx(na,nb,delay);
pmoda.a = a;
pmoda.b = b;
y = pmodsim(pmoda,e,u);

figure(1)
plot(y)
figure(2)
pmodb = newarx(na,nb,delay);
pmodb.estimParam.show=20;
pmoda.estimParam.show=20;

[pmod1,trec,stat] = estimate(pmodb,y,u);

aa = getmX(pmod1);
bb = getmX(pmoda);
disp([aa'; ...
      2*stat.stdx'; ...
      bb'])

disp(sumsqr(aa-bb))

flag = 1;
