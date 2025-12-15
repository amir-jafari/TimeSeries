function f=chkfig(n)
%CHKFIG Check to see if figure exists.
%
% CHKFIG(N)
%   N - Name of figure (string).
% Returns handle to figure with name N if it exists.
% Returns 0, otherwise.

% Yong Hu, Martin Hagan, 9-15-00
% $Revision: 1.0 $ $Date: 21-Sep-2000 14:37:36 $

%==================================================================

z = get(0,'children');
for i=1:length(z)
  typ = get(z(i),'type');
  if strcmp(typ,'figure')
    nam = get(z(i),'name');
	if strcmp(nam,n)
	  f = z(i);
	  return
	end
  end
end

f = 0;
