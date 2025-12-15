function fig1 = setfig(me)
%SETFIG Set up a figure.
%
% SETFIG(N)
%   ME - Name of figure (string).
% Returns handle to figure with name ME.

% Yong Hu, Martin Hagan, 9-15-00
% $Revision: 1.0 $ $Date: 21-Sep-2000 14:37:36 $

%==================================================================

% Get figure if it exists, or create new figure
fig1 = chkfig(me);
if length(get(fig1,'children')) == 0, fig = 0; end
if (exist('fig1')&&(fig1~=0))
  figure(fig1);
  clf reset
  set(fig1,'name',me,'numbertitle','off')
else
  fig1 = figure;
  set(fig1,'name',me,'numbertitle','off')
end
