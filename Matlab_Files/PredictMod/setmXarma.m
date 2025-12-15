function [pmod] = setmXarma(pmod,X)
%SETMXARMA Set all the ARMA model parameters from the vector X
%
%	Syntax
%
%	  pmod = setmXarma(pmod,X)
%
%	Description
%
%	  This function sets the parameters for an ARMA
%	  model to a vector of values.
%
%	  PMOD = SETMXARMA(PMOD,X)
%	    PMOD - ARMA prediction model.
%	    X    - Vector of prediction model parameters.
%
%	Examples
%
%	  Here NEWARMA is used to create an ARMA model with
%	  first order polynomials.
%
%	    pmod = newarma(1,1);
%
%	  We can set the parameters to zeros as follows:
%
%	    pmod = setmXarma(pmod,zeros(2,1));
%
%	See also SETMX, GETMX, GETMXARMA.

% Yong Hu, Martin Hagan, 9-15-00
% $Revision: 1.0 $ $Date: 21-Sep-2000 14:37:36 $


start = 0;
nnc = length(pmod.c);
for i=1:nnc,
  order = length(pmod.c{i});
  pmod.c{i} = X(start+1:start+order)';
  start = start+order;
end
nnd = length(pmod.d);
for i=1:nnd,
  order = length(pmod.d{i});
  pmod.d{i} = X(start+1:start+order)';
  start = start+order;
end

%clear all other unused parameters
pmod.a = {};
pmod.b = {};
pmod.f = {};