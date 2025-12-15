function [pmod] = setmXbjtf(pmod,X)
%SETMXBFTF Set all the BJTF model parameters from the vector X
%
%	Syntax
%
%	  pmod = setmXbjtf(pmod,X)
%
%	Description
%
%	  This function sets the parameters for a Box and Jenkins Transfer
%	  Function model to a vector of values.
%
%	  PMOD = SETMXBJTF(PMOD,X)
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
%	  We can set the parameters to zeros as follows:
%
%	    pmod = setmXbjtf(pmod,zeros(5,1));
%
%	See also SETMX, GETMX, GETMXBJTF.

% Yong Hu, Martin Hagan, 9-15-00
% $Revision: 1.0 $ $Date: 21-Sep-2000 14:37:36 $

nnb = length(pmod.b);
start = 0;
for i=1:nnb,
  order = length(pmod.b{i});
  pmod.b{i} = X(start+1:start+order)';
  start = start+order;
end
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
nnf = length(pmod.f);
for i=1:nnf,
  order = length(pmod.f{i});
  pmod.f{i} = X(start+1:start+order)';
  start = start+order;
end

%clear all other unused parameters
pmod.a = {};
