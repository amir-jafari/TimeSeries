function [pmod] = setmXarx(pmod,X)
%SETMXARX Set all the ARX model parameters from the vector X
%
%	Syntax
%
%	  pmod = setmXarx(pmod,X)
%
%	Description
%
%	  This function sets the parameters for an ARX
%	  Function model to a vector of values.
%
%	  PMOD = SETMXARX(PMOD,X)
%	    PMOD - ARX prediction model.
%	    X    - Vector of prediction model parameters.
%
%	Examples
%
%	  Here NEWARX is used to create an ARX model with
%	  first order polynomials.
%
%	    pmod = newarx(1,1);
%
%	  We can set the parameters to zeros as follows:
%
%	    pmod = setmXarx(pmod,zeros(3,1));
%
%	See also SETMX, GETMX, GETMXARX.

% Yong Hu, Martin Hagan, 9-15-00
% $Revision: 1.0 $ $Date: 21-Sep-2000 14:37:36 $

start = 0;
order = length(pmod.a{1});
pmod.a{1} = X(start+1:start+order)';
start = start+order;
nnb = length(pmod.b);
for i=1:nnb,
  order = length(pmod.b{i});
  pmod.b{i} = X(start+1:start+order)';
  start = start+order;
end
pmod.c = {};

%clear all other unused parameters
pmod.d = {};
pmod.f = {};



