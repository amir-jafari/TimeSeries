function [pmod] = setmXregr(pmod,X)
%SETMXARX Set all the regression model parameters from the vector X
%
%	Syntax
%
%	  pmod = setmXregr(pmod,X)
%
%	Description
%
%	  This function sets the parameters for a regression
%	  model to a vector of values.
%
%	  PMOD = SETMXREGR(PMOD,X)
%	    PMOD - Linear regression model.
%	    X    - Vector of prediction model parameters.
%
%	Examples
%
%	  Here NEWREGR is used to create a linear regression
%	  model with four input variables.
%
%	    pmod = newregr(4);
%
%	  We can set the parameters to zeros as follows:
%
%	    pmod = setmXregr(pmod,zeros(5,1));
%
%	See also SETMX, GETMX, GETMXREGR.

% Yong Hu, Martin Hagan, 9-15-00
% $Revision: 1.0 $ $Date: 21-Sep-2000 14:37:36 $

pmod.b{1} = X(1:length(pmod.b{1}))';

%clear all other unused parameters
pmod.a = {};
pmod.c = {};
pmod.d = {};
pmod.f = {};







