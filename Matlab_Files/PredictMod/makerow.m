function yr = makerow(y)
%MAKEROW Convert vector into row vector
%
%   Syntax
%
%     yr = makerow(y)
%
%   Description
%
%     MAKEROW converts any vector into a row vector.
%
%     MAKEROW(Y) takes this input
%       Y - Row or column vector
%     and returns,
%       YR - Row vector.
%
%   Examples
%
%     This code creates a column vector y and converts it to a row.
%
%       y = [1;2;3;4];
%       yr = makerow(y);
%

% Yong Hu, Martin Hagan, 9-15-00
% $Revision: 1.0 $ $Date: 21-Sep-2000 14:37:36 $

[n,m]=size(y);

if n>m,
  yr = y';
else
  yr = y;
end
