function [flag] = testregr()
%TESTREGR Test Regression model.

% Yong Hu, Martin Hagan, 9-15-00
% $Revision: 1.0 $ $Date: 21-Sep-2000 14:37:36 $

%Set parameters
nb = 3;
b = {[1.2 0.5 2.0 0.8]};

u=randn(nb,2000);
e=randn(1,2000)*.5;

pmoda = newregr(nb);
pmoda.b = b;
y = pmodsim(pmoda,e,u);

figure(1)
plot(y)
figure(2)
pmodb = newregr(nb);
pmodb.estimParam.show=20;
pmoda.estimParam.show=20;

[pmod1,trec,stat] = estimate(pmodb,y,u);

aa = [pmod1.b{1}];
bb = [pmoda.b{1}];
disp([aa; ...
       2*stat.stdx';...
    bb])

disp(sumsqr(aa-bb))
a=aa-bb;
%disp(sum(sum(a.*a)));

flag = 1;
