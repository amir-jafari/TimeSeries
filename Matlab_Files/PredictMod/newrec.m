function tr=newrec(epochs,varargin)
%NEWTR New training record with any number of optional fields.
%
%	Syntax
%
%	  tr = newtr(epochs,'fieldname1','fieldname2',...)
%	  tr = newtr([firstEpoch epochs],'fieldname1','fieldname2',...)
%

% Yong Hu, Martin Hagan, 9-15-00
% $Revision: 1.0 $ $Date: 21-Sep-2000 14:37:36 $

if nargin < 1,error('Not enough input arguments.'),end

names = varargin;
tr.epoch = 0:epochs;
blank = zeros(1,epochs+1)+NaN;
for i=1:length(names)
  eval(['tr.' names{i} '=blank;']);
end

