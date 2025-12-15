function [yd] = sdiff(y,d,p)
%SDIFF Seasonal differencing
%	
%	Syntax
%
%	  [yd] = sdiff(y,d,p)
%
%	Description
%	
%	  SDIFF differences the data a specified number of 
%	  times.  Each difference involves subtracting each 
%	  original time point from the point one period away.
%     
%	
%	  SDIFF(Y,D,P) takes these inputs,
%	    Y - 1xN vector containing the input sequence.
%	    D - Number of differences to take.
%	    P - Period of the difference; default = 1.
%	  and returns,
%	    YD  - RxQ vector containing the differenced series
%		
%	Examples
%
%	  Here is how to difference a sequence twice
%	  at a period of 4.
%	
%	    y=[1:20].^2;
%	    yd=sdiff(y,2,4);
%
%	Algorithm
%
%	    yd = yd(1+i*p:num_pts) - yd(1:num_pts-i*p);
%
%	See also DIFF.

% Yong Hu, Martin Hagan, 9-15-00
% $Revision: 1.0 $ $Date: 21-Sep-2000 14:37:36 $


if nargin < 2
  error('Wrong number of arguments.');
end
if nargin <3
  p=1;
end

% Make y into row format
y = makerow(y);
[m,num_pts] = size(y);

if num_pts<=(d*p),
  error('d*p is larger than the number of points.');
end

if p==1
  yd = diff(y,d,2);
else
  yd = y;
  for i=1:d,
    yd = yd(:,1+p:num_pts) - yd(:,1:num_pts-p);
    num_pts = num_pts-p;
  end
end
