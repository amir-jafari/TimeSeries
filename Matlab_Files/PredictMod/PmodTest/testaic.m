function [bic,aic] = testaic()
%TESTAIC test AIC and BIC

% Yong Hu, Martin Hagan, 9-15-00
% $Revision: 1.0 $ $Date: 21-Sep-2000 14:37:36 $

e = randn(1,2000);
u = randn(1,2000);
y = filter([1 1],[1 .5],u) + filter([1 .8],[1 -.8],e);

for i=0:5,
  pmod = newbjtf(1,1,i,1,0);
  pmod.estimParam.epochs = 50;
  pmod.estimParam.goal = 0.01;
  pmod = estimate(pmod,y,u);
  bic(i+1) = pmodbic(pmod,y,u);
  aic(i+1) = pmodaic(pmod,y,u);
end
