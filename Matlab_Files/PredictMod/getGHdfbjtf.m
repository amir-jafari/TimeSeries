function [ng,dg,nh,dh] = getGHbjtf(pmod)
%GETGH Find the G and H transfer functions for the prediction model.
%
%	Syntax
%
%	  [NG,DG,NH,DH] = GETGHBJTF(PMOD)
%
%	Description
%
%	  GETGHBJTF gets the G and H transfer functions for the Box and Jenkins
%	  Transfer Funcion prediction model PMOD.
%
%	  GETGHBJTF(PMOD) takes,
%	    PMOD - Prediction model.
%	  and returns,
%	    NG - Numerator of G transfer function.
%	    DG - Denominator of G transfer function.
%	    NH - Numerator of H transfer function.
%	    DH - Denominator of H transfer function.
%
%	Examples
%
%	  Here NEWBJTF is used to create a Box and Jenkins Transfer Function
%	  model with first order polynomials.
%
%	    pmod = newbjtf(1,1,1,1);
%
%	  The parameters of the model are all set to random values.
%	  The G and H transfer functions can be extracted:
%
%	    [ng,dg,nh,dh] = getGHbjtf(pmod);
%	    
%	Algorithm
%
%	  GETGHBJTF rearranges the system model into the form
%
%	    y(t) = sumi{Gi ui(t)} + H e(t)
%
%	See also GETGH, GETGHARX, GETGHARMA, GETGHARMAX.

% Yong Hu, Martin Hagan, 9-15-00
% $Revision: 1.0 $ $Date: 21-Sep-2000 14:37:36 $

num_inputs = length(pmod.b);
for i=1:num_inputs,
   ng{i} = [zeros(1,pmod.delay(i)) pmod.b{i}];
   if isempty(pmod.f{i})
      dg{i} = 1;
   else
      dg{i} = [1 pmod.f{i}];
   end
end


if isempty(pmod.c)
   nh = 1;
else
   nh = [1 pmod.c{1}];
end
if isempty(pmod.d)
   dh = 1;
else
   dh = [1 pmod.d{1}];
end

% Incorporate the differencing
diffop = [1 -1];
if pmod.diff(1)==0,
    ddh = 1;
else
    ddh = diffop;
    for i=2:pmod.diff(1)
        ddh = conv(ddh,diffop);
    end
end
dh = conv(dh,ddh);
 
lp = length(pmod.period);
for i=1:lp,
  per = pmod.period(i);
  ctot = per*length(pmod.c{i+1});
  nh1 = zeros(1,ctot);
  nh1(per:per:ctot) = pmod.c{i+1};
  nh1 = [1 nh1];
  nh = conv(nh,nh1);
  dtot = per*length(pmod.d{i+1});
  dh1 = zeros(1,dtot);
  dh1(per:per:dtot) = pmod.d{i+1};
  dh1 = [1 dh1];
  dh = conv(dh,dh1);
  % Incorporate the differencing
  diffop = [1 zeros(1,per)];
  diffop(end) = -1;
  if pmod.diff(i+1)==0,
    ddh = 1;
  else
    ddh = diffop;
    for i=2:pmod.diff(i+1)
        ddh = conv(ddh,diffop);
    end
  end
  dh = conv(dh,ddh);

end
