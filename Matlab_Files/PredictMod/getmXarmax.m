function [X] = getmXarmax(pmod)
%GETMXARMAX Get the parameter vector X from the ARMAX model
%
%	Syntax
%
%	  X = getmX(pmod)
%
%	Description
%
%	  This function gets the parameters for an armax transfer
%	  function model as a vector of values.  This function is normally
%	  called from GETMX
%
%	  X = GETMXARMAX(PMOD)
%	    PMOD - armax prediction model.
%	    X    - Vector of prediction model parameters.
%
%	Examples
%
%	  Here NEWARMAX is used to create an armax
%	  model with first order polynomials.
%
%	    pmod = newarmax(1,1,1);
%
%	  We can get its parameter values as follows:
%
%	    pmod.a{1}
%	    pmod.b{1}
%	    pmod.c{1}
%
%	  We can get these values as a single vector as follows:
%
%	    x = getmXarmax(pmod);
%
%	See also GETMX, SETMX, SETMXBJTF.

X = [];

X = [X; pmod.a{1}'];
nnb = length(pmod.b);
start = 0;
for i=1:nnb,
  X = [X; pmod.b{i}'];
end
X = [X; pmod.c{1}'];


