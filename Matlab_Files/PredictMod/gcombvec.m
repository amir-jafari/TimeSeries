function y = gcombvec(a1,a2,a3,a4,a5,a6,a7,a8,a9,a10)
%GCOMBVEC Generalized vector combinations.
%
%	Syntax
%
%	  gcombvec(a1,a2)
%
%	Description
%
%	  GCOMBVEC(A1,A2) takes two inputs,
%	    A1 - Matrix of N1 (column) vectors.
%	    A2 - Matrix of N2 (column) vectors.
%	  and returns a matrix of N1*N2 column vectors, where the columns
%	  consist of all possibilities of A2 vectors, appended to
%	  A1 vectors. It can handle the case in which rows of A1 is 
%	  greater than columns of A1.
%
%	Example
%	
%	  a1 = [7; 9];
%	  a2 = [1 2 3; 4 5 6];
%	  a3 = gcombvec(a1,a2)

% Revised from \toolbox\nnet\combvec.m
% Yong Hu, Martin Hagan, 9-15-00
% $Revision: 1.0 $ $Date: 21-Sep-2000 14:37:36 $

if nargin == 1
  y = a1;
elseif nargin == 2
  %len1 = length(a1);
  %len2 = length(a2);
  len1 = size(a1,2);
  len2 = size(a2,2);
  y = [nncpy(a1,len2); nncpyi(a2,len1)];
elseif nargin == 3
  y = gcombvec(gcombvec(a1,a2),a3);
elseif nargin == 4
  y = gcombvec(gcombvec(a1,a2),gcombvec(a3,a4));
elseif nargin == 5
  y = gcombvec(gcombvec(a1,a2),gcombvec(a3,a4,a5));
elseif nargin == 6
  y = gcombvec(gcombvec(a1,a2),gcombvec(a3,a4,a5,a6));
elseif nargin == 7
  y = gcombvec(gcombvec(a1,a2),gcombvec(a3,a4,a5,a6,a7));
elseif nargin == 8
  y = gcombvec(gcombvec(a1,a2),gcombvec(a3,a4,a5,a6,a7,a8));
elseif nargin == 9
  y = gcombvec(gcombvec(a1,a2),gcombvec(a3,a4,a5,a6,a7,a8,a9));
elseif nargin == 10
  y = gcombvec(gcombvec(a1,a2),gcombvec(a3,a4,a5,a6,a7,a8,a9,a10));
end

%=========================================================
function b = nncpy(m,n)

[mr,mc] = size(m);
b = zeros(mr,mc*n);
ind = 1:mc;
for i=[0:(n-1)]*mc
  b(:,ind+i) = m;
end
%=========================================================

function b = nncpyi(m,n)

[mr,mc] = size(m);
b = zeros(mr*n,mc);
ind = 1:mr;
for i=[0:(n-1)]*mr
  b(ind+i,:) = m;
end
b = reshape(b,mr,n*mc);
