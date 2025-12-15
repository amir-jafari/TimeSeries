function [pacf,phi,sigma]=parcor(acf,np)
%PARCOR Calculate the partial autocorrelation function
%	
%	Syntax
%
%	  [pacf,phi,sigma]=parcor(acf,np)
%
%	Description
%	
%	  PARCOR uses the Levinson algorithm to compute the partial
%	  autocorrelation function of the sequence y.
%	
%	  PARCOR(NA,NP) takes these inputs,
%	    ACF - Autocorrelation sequence (zero lag in the center).
%	    NP  - Number of partial autocorrelation terms to compute.
%	  and returns,
%	    PACF  - Partial autocorrelation function.
%	    PHI   - Autoregressive parameters.
%	    SIGMA - Residual variance for autoregressive model.
%		
%	Examples
%
%	  This code generates an autoregressive sequence.
%	
%	    e = randn(1,2000);
%	    y = filter(1,[1 -.8],e);
%
%	  The following commands generate the partial autocorrelation 
%	  function.  The pacf will be computed from order 1 to order 10.
%
%	    acf = xcorr(y,y,20,'unbiased');
%	    [pacf,phi,sigma]=parcor(acf,10);
%

% Yong Hu, Martin Hagan, 9-15-00
% $Revision: 1.0 $ $Date: 21-Sep-2000 14:37:36 $

% Make sure that acf is a row
acf = makerow(acf);
len=length(acf);

% Take off the negative lags of the acf (assume zero lag is in the middle)
acf = acf((len-1)/2+1:len);

% Compute the partial autocorrelation function
sigma=acf(1);
phi=[1];
q=acf(2);
for i=1:np,
  pacf(i)=-q/sigma;
  phi=[phi;0]+pacf(i)*[0;phi(i:-1:1)];
  q=acf(i+2:-1:2)*phi;
  sigma=sigma*(1-pacf(i)^2);
end
