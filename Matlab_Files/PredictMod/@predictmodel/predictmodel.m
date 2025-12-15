function pmod = predictmodel(type,na,nb,nc,nd,nf,delay,diff,per, ...
  eFcn,indexFcn,initFcn,upre,ypre,ypost)
%PREDICTMODEL  Create a custom prediction model.
%
%	Syntax
%
%	  pmod = predictmodel
%	  pmod = predictmodel(type,na,nb,nc,nd,nf,delay,diff,per,
%	    eFcn,indexFcn,initFcn,upre,ypre,ypost)
%
%	Description
%
%	  PREDICTMODEL takes these optional arguments (shown with default values):
%	    type - predict model type, 'bjtf'.
%	    na    = [na1 na2...naNI].
%	      nai   - Order of the a polynomial for input i, [].
%	    nb    = [nb1 nb2...nbNI].
%	      nbi   - Order of the b polynomial for input i, [].
%	    nc    = [nc1 nc2...ncNP]
%	      nci   - Order of the c polynomial for period i, [].
%	    nd    = [nd1 nd2...ndNP]
%	      ndi   - Order of the d polynomial for period i, [].
%	    nf    = [nf1 nf2...nfNI]
%	      nfi   - Order of the f polynomial for input i, [].
%	    delay = [delay1 delay2...delayNI], [].
%	      delayi- Pure delay between the ith input and the output.
%	    diff  = [diff1 diff2...diffNP], [].
%	      diffi - Order of the differencing for period i.
%	    per   = [per1 per2...perNP], [].
%	      peri  - Period i.
%	    eFcn  - Estimation function, 'estimlm'.
%	    indexFcn  - Performance index function, 'pmodmse'.
%	    initFcn   - Parameters initialization fucntion, 'initzero'.
%	    upre  = {upre1 upre2...upreN}, {}.
%	      uprei - ith preprocessing function for u.
%	    ypre  = {ypre1 ypre2...ypreM}, {}.
%	      yprei - ith preprocessing function for y.
%	    ypost = {ypost1 ypost2...ypostM}, {}.
%	      yposti- ith postprocessing function for y.
%	  and returns,
%	    PMOD    - Prediction model.
%
%	  The preprocessing functions yprei can be any invertible, differentiable 
%	  function such as LOG, EXP, etc.
%
%	  The estimation function eFcn can be any of the optimization
%	  functions such as ESTIMLM, ESTIMBFG, ESTIMCGP, ESTIMCGF, etc.
%
%	  The performance index function can be any of the differentiable performance
%	  functions such as PMODMSE.
%
%	Examples
%
%	  Here is how the code to create a predition model with default values,
%	  and then set its na and nb to 1 and 2, respectively.
%
%	    pmod = predictmodel;
%	    pmod.type = 'arx';
%	    pmod.na = 1;
%	    pmod.nb = 2;
%
%	  Here is the code to create the same model with one line of code.
%
%	    pmod = predictmodel('arx',1,2)
%
%	  Here the parameters of the model are estimated for up to 50 epochs 
%	  to a error goal of 0.01. The model is then used for prediction.
%	  
%	    pmod.estimParam.epochs = 50;
%	    pmod.estimParam.goal = 0.01;
%	    pmod = estimate(pmod,y,u);
%	    y2 = predict(pmod,y,u);
%	    ind = 1:length(y2);
%	    plot(ind,y,'o',ind,y2,'x')
%
%	Algorithm
%
%	  The parameters of the model are all set to zero.  They can be
%	  set to random values using the setmX function, as in
%
%	    pmod = setmX(pmod,randn(5,1));
%
%	See also SETMX

%  07-20-00
%  $Revision: 1.0 $

%SPECIAL CASE - Remaking a predict model from a struct
if (nargin == 1) & isa(type,'struct')
  pmod = class(type,'predictmodel');
  return
end

% DEFAULT ARGUMENTS
if nargin < 1, type = 'bjtf'; end

%check model type
if ~ischar(type)
  error(sprintf('"type" must be '''' .'));
end
if length(type) & ~exist(['pred' type])
  error(sprintf('"type" cannot be set to non-existing type "%s".',typ));
end

if nargin < 2, na = 0; end
if nargin < 3, nb = 0; end
if nargin < 4, nc = 0; end
if nargin < 5, nd = 0; end
if nargin < 6, nf = 0; end
if nargin < 7, delay = []; end
if nargin < 8, diff = []; end
if nargin < 9, per = []; end
if nargin < 10, eFcn = 'estimlm'; end
if nargin < 11, indexFcn = 'pmodmse'; end
if nargin < 12, initFcn = 'initzero'; end
if nargin < 13, upre = {}; end
if nargin < 14, ypre = {}; end
if nargin < 15, ypost = {}; end

% NULL PREDICT MODEL
pmod.type = type;
pmod.a = cell(1,na);
pmod.b = cell(1,nb);
pmod.c = cell(1,nc);
pmod.d = cell(1,nd);
pmod.f = cell(1,nf);
pmod.delay = delay;
pmod.diff = diff;
pmod.period = per;
pmod.estimFcn = eFcn;
pmod.estimParam = feval(eFcn,'pdefaults');
pmod.indexFcn = indexFcn;
pmod.initFcn = initFcn;
pmod.upreproc = upre;
pmod.ypreproc = ypre;
pmod.ypostproc = ypost;

% CLASS
pmod = class(pmod,'predictmodel');

 




