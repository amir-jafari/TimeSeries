function pmod = newregr(nb,delay,eFcn,indexFcn,initFcn,upre,ypre,ypost)
%NEWREGR Create a regression model.
%
%	Syntax
%
%	  pmod = newregr(nb,delay,eFcn,indexFcn,initFcn,upre,ypre,ypost)
%
%	Description
%
%	  newregr(nb,delay,eFcn,indexFcn,initFcn,upre,ypre,ypost) takes,
%	    nb       - number of input variables.
%	    delay    = [delay1 delay2...delayNI].
%	      delayi - Pure delay between the ith input and the output, default = 0;
%	    eFcn     - Estimation function, default = 'estimlm'.
%	    indexFcn - Performance index function, default = 'pmodmse'.
%	    initFcn  - Parameter initialization function, default = 'initrand'.
%	    upre     = {upre1 upre2...upreN}, default = {}.
%	      uprei  - ith preprocessing function for u.
%	    ypre     = {ypre1 ypre2...ypreM}, default = {}.
%	      yprei  - ith preprocessing function for y.
%	    ypost    = {ypost1 ypost2...ypostM}, default = {}.
%	      yposti - ith postprocessing function for y.
%	  and returns a regression model.
%
%	  The preprocessing functions yprei can be any invertible, differentiable 
%	  function such as LOG, EXP, etc.
%
%	  The estimation function eFcn can be any of the optimization
%	  functions such as ESTIMLM, ESTIMBFG, ESTIMCGP, ESTIMCGF, etc.
%
%	  The performance index function can be any of the differentiable performance
%	  functions such as ESTMSE.
%
%	  The parameter initialization function can be any of the initilization
%	  functions such as INITRAND, INITZERO.
%
%	Examples
%
%	  Here are a few points from sample y and u sequences:
%
%	    y = [-0.1867    0.5173   -3.4980    3.0093   -3.0414    1.5948];
%	    u = [-0.4326   -1.6656    0.1253    0.2877   -1.1465    1.1909];
%
%	  Here NEWREGR is used to create a regression
%	  model with first order polynomials (three total parameters).
%
%	    pmod = newregr(1);
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
%	  The parameters of the model are all set to random values.  They can be
%	  set to zero values using the setmX function, as in
%
%	    pmod = setmX(pmod,zeros(2,1));
%
%	See also NEWBJTF, NEWARX, NEWARMA, NEWARMAX.

% Yong Hu, Martin Hagan, 9-15-00
% $Revision: 1.0 $ $Date: 21-Sep-2000 14:37:36 $

if nargin < 1
  error('Not enough input arguments')
end
if nb < 0
   error('nb must be positive integers.')
end

if nargin < 2, delay = zeros(1,nb); end
if nargin < 3, eFcn = 'estimlm'; end
if nargin < 4, indexFcn = 'pmodmse'; end
if nargin < 5, initFcn = 'initrand'; end
if nargin < 6, upre = {}; end
if nargin < 7, ypre = {}; end
if nargin < 8, ypost = {}; end

pmod.type = 'regr';
pmod.a = {};
pmod.b{1} = zeros(1,nb+1);
pmod.c = {};
pmod.d = {};
pmod.f = {};
if length(delay)~= nb,
  error('delay and nb must have the same # of terms.');
end
pmod.delay = delay;
pmod.diff = [0];
pmod.period = [];
pmod.estimFcn = eFcn;
pmod.estimParam = feval(eFcn,'pdefaults');
pmod.indexFcn = indexFcn;
pmod.initFcn = initFcn;
pmod.upreproc = upre;
pmod.ypreproc = ypre;
pmod.ypostproc = ypost;

%class generation
pmod = predictmodel(pmod);

%parameters initialization
pmod = init(pmod);
