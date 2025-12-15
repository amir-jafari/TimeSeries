function [flag] = testbjtf()
%TESTBJTF Test BJTF model.

% Yong Hu, Martin Hagan, 9-15-00
% $Revision: 1.0 $ $Date: 21-Sep-2000 14:37:36 $

%Set parameters
nb = [1 1 2];
nc = [1 2 1];
nd = [2 1 1];
nf = [1 2 1];
delay = [1 2 3];
diff = [0 0 0];
per = [3 12];
f = {[0.35] [-1  .3] [.5]};
d = {[-1 0.25] [.7] [.3]};
c = {[0.65] [.5 -0.25] [-.6]};
b = {[1 1] [2 3] [1 -1 .25]};
u=randn(3,200);
e=randn(1,200)*.5;

pmoda = newbjtf(nb,nc,nd,nf,delay,diff,per);
pmoda.f = f;
pmoda.d = d;
pmoda.c = c;
pmoda.b = b;
y = pmodsim(pmoda,e,u);

figure(1)
clf reset
plot(y)

pmodb = newbjtf(nb,nc,nd,nf,delay,diff,per);
pmodb.estimParam.show=10;
pmoda.estimParam.show=10;

[pmod1,trec,stat] = estimate(pmodb,y,u);

aa = getmX(pmod1);
bb = getmX(pmoda);
disp([aa'; ...
      2*stat.stdx'; ...
      bb'])

disp(sumsqr(aa-bb))

flag = 1;
