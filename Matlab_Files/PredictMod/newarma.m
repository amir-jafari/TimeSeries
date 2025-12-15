function pmod = newarma(nc,nd,diff,per,eFcn,indexFcn,initFcn,ypre,ypost)
%NEWBJTF Create an ARMA model.
%
%	Syntax
%
%	  pmod = newarma(nc,nd,diff,per,eFcn,indexFcn,initFcn,ypre,ypost)
%
%	Description
%
%	  newarma(nc,nd,diff,per,eFcn,iFcn,ypre,ypost) takes,
%	    nc        = [nc1 nc2...ncNP]
%	      nci     - Order of the c polynomial for period i.
%	    nd        = [nd1 nd2...ndNP]
%	      ndi     - Order of the d polynomial for period i.
%	    diff      = [diff1 diff2...diffNP], default = [0].
%	      diffi   - Order of the differencing for period i.
%	    per       = [per1 per2...perNP], default = [].
%	      peri    - Period i.
%	    eFcn      - Estimation function, default = 'estimlm'.
%	    indexFcn  - Performance index function, default = 'pmodmse'.
%	    initFcn   - Parameter initialization function, default = 'initrand'.
%	    ypre      = {ypre1 ypre2...ypreM}, default = {}.
%	      yprei   - ith preprocessing function for y.
%	    ypost     = {ypost1 ypost2...ypostM}, default = {}.
%	      yposti  - ith postprocessing function for y.
%	  and returns an ARMA prediction model.
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
%	  The parameter initialization function can be any of the initilization
%	  functions such as INITRAND, INITZERO.
%
%	Examples
%
%	  Here are a few points from sample y sequences:
%
%	    y = [-0.1867    0.5173   -3.4980    3.0093   -3.0414    1.5948];
%
%	  Here NEWARMA is used to create an ARMA 
%	  model with first order polynomials (two total parameters).
%
%	    pmod = newarma(1,1);
%
%	  Here the parameters of the model are estimated for up to 50 epochs 
%	  to a error goal of 0.01. The model is then used for prediction.
%	  
%	    pmod.estimParam.epochs = 50;
%	    pmod.estimParam.goal = 0.01;
%	    pmod = estimate(pmod,y);
%	    y2 = predict(pmod,y);
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
%	See also NEWBJTF, NEWARX, NEWARMAX, NEWREGR.

% Yong Hu, Martin Hagan, 9-15-00
% $Revision: 1.0 $ $Date: 21-Sep-2000 14:37:36 $

if nargin < 2
  error('Not enough input arguments')
end
for i=1:length(nc),
  if nc(i) < 0
    error('All nc(i) must be positive integers.')
  end
end
for i=1:length(nd),
  if nd(i) < 0
    error('All nd(i) must be positive integers.')
  end
end

if nargin < 3, diff = [0]; end
if nargin < 4, per = []; end
if nargin < 5, eFcn = 'estimlm'; end
if nargin < 6, indexFcn = 'pmodmse'; end
if nargin < 7, initFcn = 'initrand'; end
if nargin < 8, ypre = {}; end
if nargin < 9, ypost = {}; end

pmod.type = 'arma';
pmod.a = {};
pmod.b = {};

nnc = length(nc);
for i=1:nnc,
  pmod.c{i} = zeros(1,nc(i));
end
nnd = length(nd);
if nnd ~= nnc,
  error('nc and nd must have the same # of terms.')
end
for i=1:nnd,
  pmod.d{i} = zeros(1,nd(i));
end
pmod.f = {};
pmod.delay = [];
if length(diff)~= nnc,
  error('nc and diff must have the same # of terms.')
end
pmod.diff = diff;
if (length(per)+1) ~= nnc,
  error('per must have one less term than nc.')
end
pmod.period = per;
pmod.estimFcn = eFcn;
pmod.estimParam = feval(eFcn,'pdefaults');
pmod.indexFcn = indexFcn;
pmod.initFcn = initFcn;
pmod.upreproc = {};
pmod.ypreproc = ypre;
pmod.ypostproc = ypost;

%class generation
pmod = predictmodel(pmod);

%parameters initialization
pmod = init(pmod);



