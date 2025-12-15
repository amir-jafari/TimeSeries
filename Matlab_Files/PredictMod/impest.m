function [g]=impest(u,y,k)
%IMPEST Estimate the impulse response.
%	
%	Syntax
%
%	  [g]=impest(u,y,k)
%
%	Description
%	
%	  IMPEST estimates the impulse response between the
%	  sequence u and the sequence y.
%	
%	  IMPEST(U,Y,K) takes these inputs,
%	    U - Input sequence.
%	    Y - Output sequence.
%	    K - Number of lags of the impulse response to compute.
%	  and returns,
%	    G - Estimate of the impulse response function.
%		
%	Examples
%
%	  This code generates a random sequence.
%	
%	    e = randn(1,2000)*0.2;
%	    u = randn(1,2000);
%	    y = filter(1,[1 .5],u) + filter(1,[1 -.8],e);
%
%	  The following commands estimate the impulse response between 
%	  u and y.  The impulse response will be computed for 10 lags.
%
%	    g = impest(u,y,10)
%

% Yong Hu, Martin Hagan, 9-15-00
% $Revision: 1.0 $ $Date: 21-Sep-2000 14:37:36 $

% Make sure that u and y are rows
u = makerow(u);
y = makerow(y);

ru = xcorr(u,u,k,'unbiased');
ruy = xcorr(u,y,k,'unbiased'); %This was for the original version of xcorr
%ruy = xcorr(y,u,k,'unbiased');  %This is for the updated xcorr

l = k+1;
r1 = [];
for n=1:l,
   r1 = [r1;ru(l-1+n:-1:l-1+n-k)];
end
rr = ruy(l:l+k)';
g = r1\rr;
