function [pmod,trec,stat]=estimlm(pmod,y,u)
%ESTIMLM Levenberg-Marquardt algorithm for prediction model fitting.
%
%	Syntax
%	
%	  [pmod,trec,stat]=estimlm(pmod,y,u)
%
%	Description
%
%	  ESTIMLM is an algorithm for estimating the parameters of a prediction
%	  model by fitting a data set according to Levenberg-Marquardt optimization.
%
%	  ESTIMLM(PMOD,Y,U) takes these inputs,
%	    PMOD - Prediction model.
%	    Y    - Prediction model desired outputs.  Y may or may not be a
%	           structure.  If Y is a structure, then Y.Y is the prediction 
%	           model desired outputs, and Y.M is the vector
%	           containing the weighting factors for each error.  
%	    U    - Prediction model inputs.
%	  and returns,
%	    PMOD - New prediction model.
%	    TREC - Training record (index).
%	            TREC.index - Training performance index.
%	            TREC.mu    - Adaptive mu value.
%	    STAT - Statistics for final model.
%               STAT.sigma - Residual variance.
%               STAT.stdx - Vector of standard deviations of parameter estimates.
%
%	  Training occurs according to the ESTIMLM's estimation parameters
%	  shown here with their default values:
%	    pmod.estimParam.epochs      10  Maximum number of epochs 
%	    pmod.estimParam.goal         0  Performance index goal
%	    pmod.estimParam.max_fail     5  Maximum validation failures
%	    pmod.estimParam.mem_reduc    1  Factor to use for memory/speed trade off.
%	    pmod.estimParam.mu       0.001  Initial mu value
%	    pmod.estimParam.mu_dec     0.1  mu decrement value
%	    pmod.estimParam.mu_inc      10  mu increment value
%	    pmod.estimParam.mu_max    1e10  Maximum mu value
%	    pmod.estimParam.min_grad  1e-4  Minimum performance gradient
%	    pmod.estimParam.show        25  Epochs between displays (NaN for no displays)
%	    pmod.estimParam.time       inf  Maximum time to train in seconds
%	    pmod.estimParam.delta     1e-7  Increment to use in computing Jacobian
%
%	Algorithm
%
%	  The Jacobian jX of error with respect to the model parameters X
%	  Is computed numerically.  Each parameter is adjusted according to 
%	  the Levenberg-Marquardt algorithm,
%
%	    jj = jX * jX
%	    je = jX * E
%	    dX = -(jj+I*mu) \ je
%
%	  where E is all errors and I is the identity matrix.
%
%	  The adaptive value MU is increased by MU_INC until the change above
%	  results in a reduced performance value.  The change is then made to
%	  the network and mu is decreased by MU_DEC.
%
%
%	  Estimation stops when any of these conditions occurs:
%	  1) The maximum number of EPOCHS (repetitions) is reached.
%	  2) The maximum amount of TIME has been exceeded.
%	  3) Performance has been minimized to the GOAL.
%	  4) The performance gradient falls below MINGRAD.
%	  5) MU exceeds MU_MAX.
%

% Yong Hu, Martin Hagan, 9-15-00
% $Revision: 1.0 $ $Date: 21-Sep-2000 14:37:36 $

% FUNCTION INFO
% =============

if ischar(pmod)
  switch (pmod)
    case 'pdefaults',
      estimParam.epochs = 100;
      estimParam.goal = 0;
      estimParam.min_grad = 1e-4;
      estimParam.mu = 0.001;
      estimParam.mu_dec = 0.1;
      estimParam.mu_inc = 10;
      estimParam.mu_max = 1e10;
      estimParam.show = 10;
      estimParam.time = inf;
      estimParam.delta = 1e-7;
	  pmod = estimParam;
    otherwise,
	  error('Unrecognized code.')
  end
  return
end

[ystru,y,m] = sepym(y);

% CALCULATION
% ===========

% Constants
uflag = nargin>2;
this = 'ESTIMLM';
epochs = pmod.estimParam.epochs;
goal = pmod.estimParam.goal;
min_grad = pmod.estimParam.min_grad;
mu = pmod.estimParam.mu;
mu_inc = pmod.estimParam.mu_inc;
mu_dec = pmod.estimParam.mu_dec;
mu_max = pmod.estimParam.mu_max;
show = pmod.estimParam.show;
time = pmod.estimParam.time;
delta = pmod.estimParam.delta;

stop = '';
startTime = clock;
X = getmX(pmod);
numParameters = length(X);
ii = sparse(1:numParameters,1:numParameters,ones(1,numParameters));
if uflag,
  [index] = calcindex(pmod,ystru,u);
else
  [index] = calcindex(pmod,ystru);
end

trec = newrec(epochs,'index','mu');

repeat = 0;

% Train
for epoch=0:epochs

  % Jacobian
  if uflag
    [je,jj,normgX]=jacobian(pmod,delta,ystru,u);
  else
    [je,jj,normgX]=jacobian(pmod,delta,ystru);
  end
  
  % Training Record
  epochPlus1 = epoch+1;
  trec.index(epoch+1) = index;
  trec.mu(epoch+1) = mu;
  
  % Stopping Criteria
  currentTime = etime(clock,startTime);
  if (index <= goal)
    stop = 'Performance goal met.';
  elseif (epoch == epochs)
    stop = 'Maximum epoch reached, Performance goal was not met.';
  elseif (currentTime > time)
    stop = 'Maximum time elapsed, performance goal was not met.';
  elseif (normgX < min_grad)
    stop = 'Minimum gradient reached, performance goal was not met.';
  elseif (mu > mu_max)
    stop = 'Maximum MU reached, performance goal was not met.';
  end
  
  % Progress
  if isfinite(show) & (~rem(epoch,show) | length(stop))
    fprintf(this);
	if isfinite(epochs) fprintf(', Epoch %g/%g',epoch, epochs); end
	if isfinite(time) fprintf(', Time %g%%',currentTime/time/100); end
	if isfinite(goal) fprintf(', %s %g/%g',upper(pmod.indexFcn),index,goal); end
	if isfinite(min_grad) fprintf(', Gradient %g/%g',normgX,min_grad); end
	if isfinite(mu_max) fprintf(', mu %g/%g',mu,mu_max); end
	fprintf('\n')
	plotindex(trec,goal,this,epoch)
    if length(stop) fprintf('%s, %s\n\n',this,stop); end
  end
 
  % Stop when criteria indicate its time
  if length(stop)
    break
  end
  
  % Levenberg Marquardt
  repeat = 0;
  while (mu <= mu_max)
    repeat = repeat + 1;
    dX = -(jj+ii*mu) \ je;
    X2 = X + dX;
    pmod2 = setmX(pmod,X2);
    if uflag,
 	  [index2] = calcindex(pmod2,ystru,u);
    else
 	  [index2] = calcindex(pmod2,ystru);
    end
	if (index2 < index)
	  X = X2; pmod = pmod2;
	  index = index2;
      mu = mu * mu_dec;
	  break
	end
	mu = mu * mu_inc;
  end
  if (mu<1e-15)
    mu = 1e-15;
  end
end

if uflag,
  e = y - predict(pmod,y,u);
else
  e = y - predict(pmod,y);
end

stat.sigma = sum(sum(e.^2)) / prod(size(e));
stat.stdx = sqrt(stat.sigma*diag(inv(jj)));

% Finish
trec = cliprec(trec,epoch);



