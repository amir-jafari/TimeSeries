function [ng,dg,nh,dh] = getGH(pmod)
%GETGH Find the G and H transfer functions for the prediction model.
%
%	Syntax
%
%	  [NG,DG,NH,DH] = GETGH(PMOD)
%
%	Description
%
%	  GETGH gets the G and H transfer functions for the prediction
%	  model PMOD.
%
%	  GETGH(PMOD) takes,
%	    PMOD - Prediction model.
%	  and returns,
%	    NG - Numerator of G transfer function.
%	    DG - Denominator of G transfer function.
%	    NH - Numerator of H transfer function.
%	    DH - Denominator of H transfer function.
%
%	  GETGH uses the prediction model type, PMOD.type, to determine
%	  how to extract the G and H transfer functions from the model.
%
%	Examples
%
%	  Here NEWBJTF is used to create a Box and Jenkins Transfer Function
%	  model with first order polynomials.
%
%	    pmod = newbjtf(1,1,1,1);
%
%	  The parameters of the model are all set to random vaules.  
%	  The G and H transfer functions can be extracted:
%
%	    [ng,dg,nh,dh] = getGH(pmod);
%	    
%	Algorithm
%
%	  GETGH rearranges the system model into the form
%
%	    y(t) = sumi{Gi ui(t)} + H e(t)
%
%	See also PREDICT, ESTIMATE

% Yong Hu, Martin Hagan, 9-15-00
% $Revision: 1.0 $ $Date: 21-Sep-2000 14:37:36 $

getGHFcn = ['getGH' pmod.type];
[ng,dg,nh,dh] = feval(getGHFcn,pmod);
