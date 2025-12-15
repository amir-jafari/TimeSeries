function [X] = getmX(pmod)
%GETMX Get the parameter vector X from the prediction model
%
%	Syntax
%
%	  X = getmX(pmod)
%
%	Description
%
%	  This function gets the parameters for a prediction model as
%	  a vector of values.
%
%	  X = GETMX(PMOD)
%	    PMOD - Prediction model.
%	    X    - Vector of prediction model parameters.
%
%	  GETMX uses the prediction model type, PMOD.type, to determine
%	  how to extract the parameters from the model.
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
%	    x = getmX(pmod);
%
%	See also GETMXARMA, GETMXARX, GETMXARMAX,GETMXBJTF, SETMX.

% Yong Hu, Martin Hagan , 9-15-00
% $Revision: 1.0 $ $Date: 21-Sep-2000 14:37:36 $

getFcn = ['getmX' pmod.type];
X = feval(getFcn,pmod);
