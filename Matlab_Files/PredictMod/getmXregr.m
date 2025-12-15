function [X] = getmXregr(pmod)
%GETMXARX Get the parameter vector X from the regression model
%
%	Syntax
%
%	  X = getmXregr(pmod)
%
%	Description
%
%	  This function gets the parameters for a regression
%	  model as a vector of values.  This function is normally
%	  called from GETMX
%
%	  X = GETMXREGR(PMOD)
%	    PMOD - regression prediction model.
%	    X    - Vector of prediction model parameters.
%
%	Examples
%
%	  Here NEWREGR	is used to create a linear regression
%	  model with one input variable.
%
%	    pmod = newregr(1);
%
%	  We can get its parameter values as follows:
%
%	    pmod.b{1}
%
%	  We can get these values as a single vector as follows:
%
%	    x = getmXregr(pmod);
%
%	See also GETMX, SETMX, SETMXBJTF.

% Yong Hu, Martin Hagan, 9-15-00
% $Revision: 1.0 $ $Date: 21-Sep-2000 14:37:36 $

X = pmod.b{1}';



