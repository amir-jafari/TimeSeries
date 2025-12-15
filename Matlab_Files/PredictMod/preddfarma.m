function [yhat] = preddfarma(pmod,y)
%PREDICT Compute one-step predictions for the ARMA model.
%
%	Syntax
%
%	  [YHAT] = PREDARMA(PMOD,Y)
%
%	Description
%
%	  PREDARMA computes the one-step-ahead predition errors for an 
%	  autoregressive moving average prediction model PMOD.
%
%	  PREDARMA(PMOD,Y) takes,
%	    PMOD - Prediction model.
%	    Y    - Prediction model desired outputs.
%	  and returns,
%	    YHAT - One-step-ahead predictions.
%
%
%	Examples
%
%	  Here are a few points from sample y sequence:
%
%	    y = [-0.1867    0.5173   -3.4980    3.0093   -3.0414    1.5948];
%
%	  Here NEWARMA is used to create an ARMA
%	  model with first order polynomials.
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
%	  PREDARMA assumes the system model is in the form
%
%	    y(t) =  H e(t)
%
%	  The prediction is computed as:
%
%	    yhat(t) = [1-inv(H)] y(t)
%
%	See also PREDICT, PREDARX, PREDARMAX, PREDBJTF, PREDREGR.

% Yong Hu, Martin Hagan, 9-15-00
% $Revision: 1.0 $ $Date: 21-Sep-2000 14:37:36 $

% Expand the parameter vectors into g and h form
[ng,dg,nh,dh] = getGHdf(pmod);

% Make the numerator and denominator of h the same size.
num_inputs = length(pmod.b);
lnh = length(nh);
ldh = length(dh);
if lnh > ldh
  dh = [dh zeros(1,lnh-ldh)];
elseif ldh > lnh
  nh = [nh zeros(1,ldh-lnh)];
end

% Compute the prediction.
yhat=filter((nh-dh),nh,y);
