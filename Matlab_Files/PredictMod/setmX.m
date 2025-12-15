function [pmod] = setmX(pmod,X)
%SETMX Set all the prediction model parameters using the vector X
%
%	Syntax
%
%	  pmod = setmX(pmod,X)
%
%	Description
%
%	  This function sets the parameters for a prediction model to
%	  a vector of values.
%
%	  PMOD = SETMX(PMOD,X)
%	    PMOD - Prediction model.
%	    X    - Vector of prediction model parameters.
%
%	  SETMX uses the prediction model type, PMOD.type, to determine
%	  how to place the parameters in the model.
%
%	Examples
%
%	  Here NEWBJTF is used to create a Box and Jenkins Transfer Function
%	  model with first order polynomials.  All parameters are set to 
%	  random values by default. 
%
%	    pmod = newbjtf(1,1,1,1);
%
%	  We can set the parameters to zeros as follows:
%
%	    pmod = setmX(pmod,zeros(5,1));
%
%	See also GETMX.

% Yong Hu, Martin Hagan, 9-15-00
% $Revision: 1.0 $ $Date: 21-Sep-2000 14:37:36 $

setFcn = ['setmX' pmod.type];
pmod = feval(setFcn,pmod,X);
