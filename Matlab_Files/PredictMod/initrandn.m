function pmod=initrandn(pmod)
%INIT Initialize a prediction model parameters to 
%  normally distributed random value.
%
%	Syntax
%
%	  PMOD = INITRANDN(PMOD)
%
%	Description
%
%	  INITRANDN(PMOD) returns prediction model PMOD with a,b,c,d,f values
%	  updated to normally distributed random value.
%
%	  INITRAND(PMOD) takes,
%	    PMOD - Prediction model.
%	  and returns,
%	    PMOD - Updated prediction model.
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
%	  INITRANDN reinitializes those parameters to normally distributed
%	  random values.
%
%	    pmod.initFcn = 'initrandn';
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
%	See also INITZERO, INITRAND.

% Yong Hu, Martin Hagan, 9-15-00
% $Revision: 1.0 $ $Date: 21-Sep-2000 14:37:36 $

if nargin < 1
  error('Not enough input arguments')
end

variance = 0.125;

for i=1:length(pmod.a)
   pmod.a{i} = variance*(randn(size(pmod.a{i})));
end

for i=1:length(pmod.b)
   pmod.b{i} = variance*(randn(size(pmod.b{i})));
end

for i=1:length(pmod.c)
   pmod.c{i} = variance*(randn(size(pmod.c{i})));
end

for i=1:length(pmod.d)
   pmod.d{i} = variance*(randn(size(pmod.d{i})));
end

for i=1:length(pmod.f)
   pmod.f{i} = variance*(randn(size(pmod.f{i})));
end


