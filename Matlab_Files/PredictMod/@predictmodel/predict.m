function [yhat] = predict(pmod,y,u)
%PREDICT Compute one-step predictions.
%
%	Syntax
%
%	  [YHAT] = PREDICT(PMOD,Y,U)
%
%	Description
%
%	  PREDICT computes the one-step-ahead predition errors for a 
%	  prediction model PMOD.
%
%	  PREDICT(PMOD,Y,U) takes,
%	    PMOD - Prediction model.
%	    Y    - Prediction model desired outputs.
%	    U    - Prediction model inputs.
%	  and returns,
%	    YHAT - One-step-ahead predictions.
%
%
%	Examples
%
%	  Here are a few points from sample y and u sequences:
%
%	    y = [-0.1867    0.5173   -3.4980    3.0093   -3.0414    1.5948];
%	    u = [-0.4326   -1.6656    0.1253    0.2877   -1.1465    1.1909];
%
%	  Here NEWBJTF is used to create a Box and Jenkins Transfer Function
%	  model with first order polynomials.
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
%	  PREDICT calls the function indicated by PMOD.type.  It
%	  appends "pred" to the front of PMOD.type to determine
%	  the prediction function to use.
%
%	See also PMODSIM, ESTIMATE

% Yong Hu, Martin Hagan, 9-15-00
% $Revision: 1.0 $ $Date: 21-Sep-2000 14:37:36 $

uflag = nargin>2;

predFcn = ['pred' pmod.type];

if uflag,
  yhat = feval(predFcn,pmod,y,u);
else
  yhat = feval(predFcn,pmod,y);
end
