function [ng,dg,nh,dh] = getGHarx(pmod)
%GETGH Find the G and H transfer functions for the prediction model.
%
%	Syntax
%
%	  [NG,DG,NH,DH] = GETGHARX(PMOD)
%
%	Description
%
%	  GETGHARX gets the G and H transfer functions for arx
%	  prediction model PMOD.
%
%	  GETGHARX(PMOD) takes,
%	    PMOD - Prediction model.
%	  and returns,
%	    NG - Numerator of G transfer function.
%	    DG - Denominator of G transfer function.
%	    NH - Numerator of H transfer function.
%	    DH - Denominator of H transfer function.
%
%	Examples
%
%	  Here NEWARX is used to create an armax
%	  model with first order polynomials.
%
%	    pmod = newarx(1,1);
%
%	  The parameters of the model are all set to random values.
%	  The G and H transfer functions can be extracted:
%
%	    [ng,dg,nh,dh] = getGHarx(pmod);
%	    
%	Algorithm
%
%	  GETGHARX rearranges the system model into the form
%
%	    y(t) = sumi{Gi ui(t)} + H e(t)
%
%	See also GETGH, GETGHBJTF, GETGHARMAX, GETGHARMA.

% Yong Hu, Martin Hagan, 9-15-00
% $Revision: 1.0 $ $Date: 21-Sep-2000 14:37:36 $

num_inputs = length(pmod.b);
for i=1:num_inputs,
   ng{i} = [zeros(1,pmod.delay(i)) pmod.b{i}];
   if isempty(pmod.a)
      dg{i} = 1;
   else
      dg{i} = [1 pmod.a{1}];
   end
end

nh = [1];
if isempty(pmod.a)
   dh = 1;
else
   dh = [1 pmod.a{1}];
end

