function [X] = getmXarx(pmod)
%GETMXARX Get the parameter vector X from the ARX model
%
%	Syntax
%
%	  X = getmX(pmod)
%
%	Description
%
%	  This function gets the parameters for an arx transfer
%	  function model as a vector of values.  This function is normally
%	  called from GETMX
%
%	  X = GETMXARX(PMOD)
%	    PMOD - arx prediction model.
%	    X    - Vector of prediction model parameters.
%
%	Examples
%
%	  Here NEWARX is used to create an arx
%	  model with first order polynomials.
%
%	    pmod = newarx(1,1);
%
%	  We can get its parameter values as follows:
%
%	    pmod.a{1}
%	    pmod.b{1}
%
%	  We can get these values as a single vector as follows:
%
%	    x = getmXarx(pmod);
%
%	See also GETMX, SETMX, SETMXARX.

% Yong Hu, Martin Hagan, 9-15-00
% $Revision: 1.0 $ $Date: 21-Sep-2000 14:37:36 $

X = [];

X = [X; pmod.a{1}'];
nnb = length(pmod.b);
for i=1:nnb,
  X = [X; pmod.b{i}'];
end


