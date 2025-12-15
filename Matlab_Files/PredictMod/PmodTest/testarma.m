function [flag] = testarma()
%TESTARMA Test ARMA model.

% Yong Hu, Martin Hagan, 9-15-00
% $Revision: 1.0 $ $Date: 21-Sep-2000 14:37:36 $

%Set parameters
nc = [1 2 1];
nd = [2 1 1];
diff = [0 0 0];
per = [3 12];
d = {[-1 0.25] [.7] [.3]};
c = {[0.65] [.5 -0.25] [-.6]};
e=randn(1,2000)*.5;

pmoda = newarma(nc,nd,diff,per);
pmoda.d = d;
pmoda.c = c;

pmoda.ypostproc = {'exp'};
y = pmodsim(pmoda,e);

%figure(1)
%clf reset
%plot(y)

pmodb = newarma(nc,nd,diff,per);
pmodb.estimParam.show=10;
pmoda.estimParam.show=10;

pmodb.ypreproc = {'log'};
[pmodb,trec,stat] = estimate(pmodb,y);

aa = getmX(pmodb);
bb = getmX(pmoda);
disp([aa'; ...
      2*stat.stdx'; ...
      bb'])

disp(sumsqr(aa-bb))

flag = 1;
