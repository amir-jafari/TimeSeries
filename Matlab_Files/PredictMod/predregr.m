function [yhat] = predregr(pmod,y,u)
%PREDICT Compute one-step predictions for regression model.
%
%	Syntax
%
%	  [YHAT] = PREDREGR(PMOD,Y,U)
%
%	Description
%
%	  PREDREGR computes the one-step-ahead predition errors for a 
%	  regression prediction model PMOD.
%
%	  PREDREGR(PMOD,Y,U) takes,
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
%	  Here NEWREGR is used to create a linear regression
%	  model with one input variable.
%
%	    pmod = newregr(1);
%
%	  Here the parameters of the model are estimated for up to 50 epochs 
%	  to an error goal of 0.01. The model is then used for prediction.
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
%	  PREDREGR assumes the system model is in the form
%
%	    y = pmod.b{1}*u + e
%
%	  where u is the input and e is white noise.
%	  The prediction is computed as:
%
%	    yhat = pmod.b{1}*u
%
%	See also PREDICT, PREDBJTF, PREDARMAX, PREDARX, PREDARMA

% Yong Hu, Martin Hagan, 9-15-00
% $Revision: 1.0 $ $Date: 21-Sep-2000 14:37:36 $

% Make sure that the number of inputs is correct.
num_inputs = length(pmod.b{1})-1;
[m,n] = size(u);
if m~=num_inputs,
  error('The number of rows of u does not match the number of elements in b')
end

% Compute the prediction.
for i=1:m
   if isempty(pmod.delay)
      idel = 0;
   else
      idel = pmod.delay(i);
   end
   udelay(i,:) = [zeros(1,idel) u(i,1:n-idel)];
end
u1 = [ones(1,n);udelay];
yhat = pmod.b{1}*u1;


%u = [ones(1,length(y));u];
%yhat = pmod.b{1}*u;

