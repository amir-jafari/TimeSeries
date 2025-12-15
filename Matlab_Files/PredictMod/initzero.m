function pmod=initzero(pmod)
%INIT Initialize a prediction model parameters to zero.
%
%	Syntax
%
%	  PMOD = INITZERO(PMOD)
%
%	Description
%
%	  INITZERO(PMOD) returns prediction model PMOD with non-empty a,b,c,d,f
%	  values to zero.
%
%	  INITZERO(PMOD) takes,
%	    PMOD - Prediction model.
%	  and returns,
%	    PMOD - Updated prediction model.
%
%	Examples
%
%	  Here an ARX model is created with nb and na both equal 1.0.
%	  Then A and B parameters are random values by default.
%
%	    pmod = newarx(1,1);
%	    pmod.a{1}
%	    pmod.b{1}
%
%	  INITRAND reinitializes those parameters to zeros.
%
%	    pmod.initFcn = 'initzero';
%	    pmod = init(pmod);
%	    pmod.a{1}
%	    pmod.b{1}
%
%	Algorithm
%
%	  INIT calls PMOD.initFcn to initialize the parameters.
%
%	  Typically, PMOD.initFcn is set to 'initrand' which initializes each
%	  parameter to uniformly distributed random value.
%
%	See also INITRAND, INITRANDN.

% Yong Hu, Martin Hagan, 9-15-00
% $Revision: 1.0 $ $Date: 21-Sep-2000 14:37:36 $

if nargin < 1
  error('Not enough input arguments')
end

for i=1:length(pmod.a)
   pmod.a{i} = zeros(size(pmod.a{i}));
end

for i=1:length(pmod.b)
   pmod.b{i} = zeros(size(pmod.b{i}));
end

for i=1:length(pmod.c)
   pmod.c{i} = zeros(size(pmod.c{i}));
end

for i=1:length(pmod.d)
   pmod.d{i} = zeros(size(pmod.d{i}));
end

for i=1:length(pmod.f)
   pmod.f{i} = zeros(size(pmod.f{i}));
end


