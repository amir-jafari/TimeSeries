function pmod = newbjtf(nb,nc,nd,nf,delay,diff,per,eFcn,indexFcn,initFcn,upre,ypre,ypost)
%NEWBJTF Create a Box and Jenkins Transfer Function model.
%
%	Syntax
%
%	  pmod = newbjtf(nb,nc,nd,nf,delay,diff,per,eFcn,indexFcn,initFcn,upre,ypre,ypost)
%
%	Description
%
%	  newbjtf(nb,nc,nd,nf,delay,diff,per,eFcn,indexFcn,initFcn,upre,ypre,ypost) takes,
%	    nb       = [nb1 nb2...nbNI].
%	      nbi    - Order of the b polynomial for input i.
%	    nc       = [nc1 nc2...ncNP]
%	      nci    - Order of the c polynomial for period i.
%	    nd       = [nd1 nd2...ndNP]
%	      ndi    - Order of the d polynomial for period i.
%	    nf       = [nf1 nf2...nfNI]
%	      nfi    - Order of the f polynomial for input i.
%	    delay    = [delay1 delay2...delayNI], default = [1].
%	      delayi - Pure delay between the ith input and the output.
%	    diff     = [diff1 diff2...diffNP], default = [0].
%	      diffi  - Order of the differencing for period i.
%	    per      = [per1 per2...perNP], default = [].
%	      peri   - Period i.
%	    eFcn     - Estimation function, default = 'estimlm'.
%	    indexFcn - Performance index function, default = 'estmse'.
%	    initFcn  - Parameter initialization function, default = 'initrand'.
%	    upre     = {upre1 upre2...upreN}, default = {}.
%	      uprei  - ith preprocessing function for u.
%	    ypre     = {ypre1 ypre2...ypreM}, default = {}.
%	      yprei  - ith preprocessing function for y.
%	    ypost    = {ypost1 ypost2...ypostM}, default = {}.
%	      yposti - ith postprocessing function for y.
%	  and returns a Box and Jenkins transfer function prediction model.
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
%	  Here NEWBJTF is used to create a Box and Jenkins Transfer Function
%	  model with first order polynomials (five total parameters).
%
%	    pmod = newbjtf(1,1,1,1);
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
%	    pmod = setmX(pmod,zeros(5,1));
%
%	See also NEWARX, NEWARMA, NEWARMAX, NEWREGR.

% Yong Hu, Martin Hagan, 9-15-00
% $Revision: 1.0 $ $Date: 21-Sep-2000 14:37:36 $

if nargin < 4
  error('Not enough input arguments')
end
for i=1:length(nb),
  if nb(i) < 0
    error('All nb(i) must be positive integers.')
  end
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
for i=1:length(nf),
  if nf(i) < 0
    error('All nf(i) must be positive integers.')
  end
end

if nargin < 5, delay = 1; end
if nargin < 6, diff = [0]; end
if nargin < 7, per = []; end
if nargin < 8, eFcn = 'estimlm'; end
if nargin < 9, indexFcn = 'pmodmse'; end
if nargin < 10, initFcn = 'initrand'; end
if nargin < 11, upre = {}; end
if nargin < 12, ypre = {}; end
if nargin < 13, ypost = {}; end

pmod.type = 'bjtf';
pmod.a = cell(1,0);

nnb = length(nb);
for i=1:nnb,
  pmod.b{i} = zeros(1,nb(i)+1);
end
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
nnf = length(nf);
if nnf ~= nnb,
  error('nf and nb must have the same # of terms.')
end
for i=1:nnf,
  pmod.f{i} = zeros(1,nf(i));
end
if length(delay)~= nnb,
  error('delay and nb must have the same # of terms.')
end
pmod.delay = delay;
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
pmod.upreproc = upre;
pmod.ypreproc = ypre;
pmod.ypostproc = ypost;

%class generation
pmod = predictmodel(pmod);

%parameters initialization
pmod = init(pmod);
