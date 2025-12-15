function [X] = getmXbjtf(pmod)
%GETMXBJTF Get the parameter vector X from the Box and Jenkins TF model
%
%	Syntax
%
%	  X = getmXbjtf(pmod)
%
%	Description
%
%	  This function gets the parameters for a Box and Jenkins Transfer
%	  Function model as a vector of values.  This function is normally
%	  called from GETMX
%
%	  X = GETMXBJTF(PMOD)
%	    PMOD - Box and Jenkins Transfer Function prediction model.
%	    X    - Vector of prediction model parameters.
%
%	Examples
%
%	  Here NEWBJTF is used to create a Box and Jenkins Transfer Function
%	  model with first order polynomials.
%
%	    pmod = newbjtf(1,1,1,1);
%
%	  We can get its parameter values as follows:
%
%	    pmod.b{1}
%	    pmod.c{1}
%	    pmod.d{1}
%	    pmod.f{1}
%
%	  We can get these values as a single vector as follows:
%
%	    x = getmXbjtf(pmod);
%
%	See also SETMX, GETMX, SETMXBJTF.

% Yong Hu, Martin Hagan, 9-15-00
% $Revision: 1.0 $ $Date: 21-Sep-2000 14:37:36 $

X = [];

nnb = length(pmod.b);
for i=1:nnb,
  X = [X; pmod.b{i}'];
end
nnc = length(pmod.c);
for i=1:nnc,
  X = [X; pmod.c{i}'];
end
nnd = length(pmod.d);
for i=1:nnd,
  X = [X; pmod.d{i}'];
end
nnf = length(pmod.f);
for i=1:nnf,
  X = [X; pmod.f{i}'];
end
