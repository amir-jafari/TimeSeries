function [pmod] = setmXarmax(pmod,X)
%SETMXARMAX Set all the ARMAX model parameters from the vector X
%
%	Syntax
%
%	  pmod = setmXarmax(pmod,X)
%
%	Description
%
%	  This function sets the parameters for an ARMAX
%	  Function model to a vector of values.
%
%	  PMOD = SETMXARMAX(PMOD,X)
%	    PMOD - ARMAX prediction model.
%	    X    - Vector of prediction model parameters.
%
%	Examples
%
%	  Here NEWARMAX is used to create an ARMAX model with
%	  first order polynomials.
%
%	    pmod = newarmax(1,1,1);
%
%	  We can set the parameters to zeros as follows:
%
%	    pmod = setmXarmax(pmod,zeros(4,1));
%
%	See also SETMX, GETMX, GETMXARMAX.

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
order = length(pmod.c{1});
pmod.c{1} = X(start+1:start+order)';

%clear all other unused parameters
pmod.d = {};
pmod.f = {};


