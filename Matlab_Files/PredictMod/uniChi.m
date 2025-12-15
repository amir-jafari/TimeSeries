function [pass,q,n] = uniChi(pmod,y,k,alpha)
%UNICHI Univariate Portmanteau lack-of-fit (Chi-square) test.
%	
%	Syntax
%
%	  [pass,q,n] = uniChi(pmod,y,k,alpha)
%
%	Description
%	
%	  UNICHI performs a univariate portmanteau lack-of-fit test. It calculates
%	  the chi-square statistics on the residuals of a fitted model. 
%	  Large values of the statistic indicate that the residuals are
%	  not white, and therefore the model does not provide a good fit.
%	
%	  UNICHI(PMOD,Y,K) takes these inputs,
%	    PMOD  - Prediction model.
%	    Y     - Prediction model desired outputs.
%	    K     - Number of lags of the ACF to use in the statistic, default = 20.
%	    ALPHA - Probability of type I error, default = 0.05.
%	  and returns,
%	    PASS - 1 if the chi-square test is passed, 0 otherwise.
%	    Q    - The chi-square statistic.
%	    N    - Degrees of freedom.
%		
%	Examples
%
%	  This code generates an ARMA sequence, and an estimated
%	  ARMA prediction model. 
%	
%	    e = randn(1,2000);
%	    y = filter([1 0.5],[1 -.8],e);
%	    pmod = newarma(1,1);
%	    pmod.estimParam.show = NaN;
%	    [pmod,trec,stat] = estimate(pmod,y);
%
%	  Then a portmanteau lack-of-fit test is then performed 
%    on the estimated ARMA model.
%
%	    [pass,q,n] = unichi(pmod,y);
%
%	See also MULTICHI.

% Yong Hu, Martin Hagan, 9-15-00
% $Revision: 1.0 $ $Date: 21-Sep-2000 14:37:36 $

if nargin < 2
  error('Not enough input arguments')
end
if nargin<3, k = 20; end
if nargin<4, alpha = 0.05; end


e = y - predictdf(pmod,y);
%auto correlation function of residual error
acf = xcorr(e,e,k,'unbiased');
acf(1:(length(acf)-1)/2) = [];
acf = acf/acf(1);
acf(1) = [];

q = length(y)*sum(acf.^2);
X = getmX(pmod);
numPara = length(X);
n = k-numPara;

%calculate chi-square cumulative density function
pr = chisqrdf(q,n);

if (1-pr)>alpha
   pass = 1;
else
   pass = 0;
end

