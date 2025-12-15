function tr=cliprec(tr,epochs)
%CLIPTR Clip training record to the final number of epochs.
%
%	Syntax
%
%	  tr = cliptr(tr,epochs)
%

% Yong Hu, Martin Hagan, 9-15-00
% $Revision: 1.0 $ $Date: 21-Sep-2000 14:37:36 $

indices = 1:(epochs+1);
names = fieldnames(tr);
for i=1:length(names)
  name = names{i};
  eval(['tr.' name ' = tr.' name '(:,indices);']);
end
