function pmod=initrand(pmod)
%INIT Initialize a prediction model parameters to 
%  uniformly distributed random value.
%
%	Syntax
%
%	  PMOD = INITRAND(PMOD)
%
%	Description
%
%	  INITRAND(PMOD) returns prediction model PMOD with a,b,c,d,f values
%	  updated to uniformly distributed random value ranging from -0.125 
%	  to 0.125.
%
%	  INITRAND(PMOD) takes,
%	    PMOD - Prediction model.
%	  and returns,
%	    PMOD - Updated prediction model.
%
%
%	Examples
%
%	  Here an ARX model is created with nb and na both equal 1.0.
%	  Then A and B parameters are both set to 0.
%
%	    pmod = newarx(1,1);
%	    pmod.initFcn = 'initzero';
%	    pmod = init(pmod);
%	    pmod.a{1}
%	    pmod.b{1}
%
%	  INITRAND reinitializes those parameters to uniformly distributed 
%	  random value.
%
%	    pmod.initFcn = 'initrand';
%	    pmod = init(pmod);
%	    pmod.a{1}
%	    pmod.b{1}
%
%	Algorithm
%
%	  INIT calls PMOD.initFcn to initialize the parameters
%
%	  Typically, PMOD.initFcn is set to 'initrand' which initializes each
%	  parameter to uniformly distributed random value.
%
%	See also INITZERO, INITRANDN.

% Yong Hu, Martin Hagan, 9-15-00
% $Revision: 1.0 $ $Date: 21-Sep-2000 14:37:36 $

if nargin < 1
  error('Not enough input arguments')
end

llim = -0.125;
hlim = 0.125;

range = hlim-llim;
bias = llim/range;

for i=1:length(pmod.a)
   pmod.a{i} = range*(rand(size(pmod.a{i}))+bias);
end

for i=1:length(pmod.b)
   pmod.b{i} = range*(rand(size(pmod.b{i}))+bias);
end

for i=1:length(pmod.c)
   pmod.c{i} = range*(rand(size(pmod.c{i}))+bias);
end

for i=1:length(pmod.d)
   pmod.d{i} = range*(rand(size(pmod.d{i}))+bias);
end

for i=1:length(pmod.f)
   pmod.f{i} = range*(rand(size(pmod.f{i}))+bias);
end


