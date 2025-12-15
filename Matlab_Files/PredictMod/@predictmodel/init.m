function pmod=init(pmod)
%INIT Initialize a prediction model.
%
%	Syntax
%
%	  pmod = init(pmod)
%
%	Description
%
%	  INIT(PMOD) returns prediction model PMOD with model parameters
%	  updated according to the model initialization function, indicated
%	  by pmod.initFcn.
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
%	  INIT reinitializes those parameters to random value.
%
%	    pmod.initFcn = 'initrand';
%	    pmod = init(pmod);
%	    pmod.a{1}
%	    pmod.b{1}
%
%	  The parameters are random values again, which are the initial values
%	  used by ARX prediction model (see NEWARX). 
%
%	Algorithm
%
%	  INIT calls PMOD.initFcn to initialize the parameters
%
%	  Typically, PMOD.initFcn is set to 'initrand' which initializes each
%	  parameters to random value.
%
%	See also INITRAND, INITZERO.

% 08-07-00
% $Revision: 1.0 $

%pmod = struct(pmod);
initFcn = pmod.initFcn;
pmod = feval(initFcn,pmod);
%pmod = class(pmod,'predictmodel');
