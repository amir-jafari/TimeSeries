function [X] = getmXarma(pmod)
%GETMXARMA Get the parameter vector X from the ARMA model
%
%	Syntax
%
%	  X = getmXarma(pmod)
%
%	Description
%
%	  This function gets the parameters for an ARMA
%	  model as a vector of values.  This function is normally
%	  called from GETMX
%
%	  X = GETMXARMA(PMOD)
%	    PMOD - ARMA prediction model.
%	    X    - Vector of prediction model parameters.
%
%	Examples
%
%	  Here NEWARMA is used to create an ARMA
%	  model with first order polynomials.
%
%	    pmod = newarma(1,1);
%
%	  We can get its parameter values as follows:
%
%	    pmod.c{1}
%	    pmod.d{1}
%
%	  We can get these values as a single vector as follows:
%
%	    x = getmXarma(pmod);
%
%	See also SETMX, GETMX, SETMXARMA.

% Yong Hu, Martin Hagan, 9-15-00
% $Revision: 1.0 $ $Date: 21-Sep-2000 14:37:36 $

X = [];

nnc = length(pmod.c);
for i=1:nnc,
  X = [X; pmod.c{i}'];
end
nnd = length(pmod.d);
for i=1:nnd,
  X = [X; pmod.d{i}'];
end
