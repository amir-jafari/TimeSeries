function [ystru,y,m]=sepym(y)
%SEPYM Seperate y and m from y if y is a structure.
%
%	Syntax
%
%	  [ystru,y,m]=sepym(y);
%
%	Description
%
%	  This function extracts y and m from structured y if input y is a 
%	  structure, and retains y as ystru.  If input y is not a structure,
%	  m is returned as a row vector, which has the same order as y.
%	  Constructs a structure ystru, which has field ystru.y = y and
%	  ystru.m = m.
%
%	  [ystru,y,m]=sepym(y) takes,
%	    y     - Desired prediction model output.  If y is a structure, 
%	            y contains y.y, and y.m.
%	  and returns,
%	    YSTRU - Structured y with field ystru.y and ystru.m.
%	    y     - Desired prediction model output, not a structure.  
%	    M     - Row vector containing the weighting factors for
%	            each error.  If input y is not a structure, m is an 
%	            ONE row vector which has the same order as y.
%

% Yong Hu, Martin Hagan, 9-15-00
% $Revision: 1.0 $ $Date: 21-Sep-2000 14:37:36 $




if isstruct(y) 
   ystru = y;
   if isfield(y,'m')
      m = y.m;
   else
      error('m is not a field of input y');
   end
   if isfield(y,'y')
      y = y.y;
   else
      error('y is not a field of input y');
   end
   if length(y)~=length(m)
      error('y and m should have the same length');
   end
else
   m = ones(1,length(y));
   ystru.y = y;
   ystru.m = m;
end

