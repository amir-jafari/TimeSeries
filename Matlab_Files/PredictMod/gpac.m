function [gpac_array]=gpac(acf,nrows,ncols)
%GPAC Calculate the generalized partial autocorrelation function
%	
%	Syntax
%
%	  [gpac_array] = gpac(acf,nrows,ncols)
%
%	Description
%	
%	  GPAC computes the generalized partial
%	  autocorrelation function for the acf.
%	
%	  GPAC(ACF,NROWS,NCOLS) takes these inputs,
%	    ACF   - Autocorrelation sequence (zero lag in the center).
%	    NROWS - Number of rows of the GPAC to compute.
%	    NCOLS - Number of columns of the GPAC to compute.
%	  and returns,
%	    GPAC_ARRAY - Generalized partial autocorrelation function.
%		
%	Examples
%
%	  This code generates an autoregressive sequence.
%	
%	    e = randn(1,2000);
%	    y = filter(1,[1 -.8],e);
%
%	  The following commands generate the generalized partial autocorrelation 
%	  function.  The gpac will be computed for 7 rows and 7 columns.
%
%	    acf = xcorr(y,y,20,'unbiased');
%	    [gpac_array] = gpac(acf,7,7)
%

% Yong Hu, Martin Hagan, 9-15-00
% $Revision: 1.0 $ $Date: 21-Sep-2000 14:37:36 $

gpac_array = zeros(nrows,ncols);

if (size(acf,1))~=1,
   acf = acf';
end
l = fix(length(acf)/2)+1;

for j = 1:nrows,
   for m = 1:ncols,
      r1=[];
      for n=1:m,
         r1 = [r1;acf(l+j-2+n:-1:l+j-1+n-m)];
      end
      rr=acf(j+l:j+l+m-1)';
      r2 = r1;
      r2(:,m) = rr;
      gpac_array(j,m) = det(r2)/det(r1);
   end
end
