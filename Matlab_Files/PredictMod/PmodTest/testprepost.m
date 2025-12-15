function [flag] = testprepost()
%TESTPREPOST Test pre-post processing

% Yong Hu, Martin Hagan, 9-15-00
% $Revision: 1.0 $ $Date: 21-Sep-2000 14:37:36 $

%Set parameters
nb = [1 1 2]; 
na = 2;
delay = [1 2 3];
b = {[1 1] [2 3] [1 -1 .25]};
a = {[-1 0.25]};
rand('state',0);
randn('state',0);
u=rand(3,2000);
e=randn(1,2000)*.5;

pmoda = newarx(na,nb,delay);
pmoda.a = a;
pmoda.b = b;
pmoda.upreproc = {'exp','log';'log',[];'exp',[]};
pmoda.ypostproc = {'exp'};
y = pmodsim(pmoda,e,u);

%figure(1)
%plot(y)
%figure(2)

pmodb = newarx(na,nb,delay);
pmodb.estimParam.show = 20;

%preprocessing
%u(1,:) = log(exp((u(1,:))));
%u(2,:) = exp(u(2,:));
%u(3,:) = log(u(3,:));
%y = exp(log((y)));
%assign upreproc, ypreproc
pmodb.upreproc = {'exp','log';'log',[];'exp',[]};
pmodb.ypreproc = {'log'};

[pmod1,trec,stat] = estimate(pmodb,y,u);

aa = getmX(pmod1);
bb = getmX(pmoda);
disp([aa'; ...
      2*stat.stdx'; ...
      bb'])

disp(sumsqr(aa-bb))

flag = 1;
