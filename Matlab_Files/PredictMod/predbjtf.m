function [yhat] = predbjtf(pmod,y,u)
%PREDICT Compute one-step predictions for the Box and Jenkins Transfer Function model.
%
%	Syntax
%
%	  [YHAT] = PREDBJTF(PMOD,Y,U)
%
%	Description
%
%	  PREDBJTF computes the one-step-ahead predition errors for a 
%	  Box and Jenkins Transfer Function prediction model PMOD.
%
%	  PREDBJTF(PMOD,Y,U) takes,
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
%	  PREDBJTF assumes the system model is in the form
%
%	    y(t) = sumi{Gi ui(t)} + H e(t)
%
%	  where ui(t) is the ith input and e(t) is white noise.
%	  The prediction is computed as:
%
%	    yhat(t) = [1-inv(H)] y(t) + inv(H) sum{Gi ui(t)}
%
%	See also PREDICT, PREDARX, PREDARMA, PREDARMAX, PREDREGR

% Yong Hu, Martin Hagan, 9-15-00
% $Revision: 1.0 $ $Date: 21-Sep-2000 14:37:36 $

% Expand the parameter vectors into g and h form
[ng,dg,nh,dh] = getGH(pmod);

% Make the numerator and denominator of h the same size.
num_inputs = length(pmod.b);
lnh = length(nh);
ldh = length(dh);
if lnh > ldh
  dh = [dh zeros(1,lnh-ldh)];
elseif ldh > lnh
  nh = [nh zeros(1,ldh-lnh)];
end

% Make sure that the number of inputs is correct.
[m,n] = size(u);
if m~=num_inputs,
  error('The number of rows of u does not match the number of cells in b')
end

% Compute the prediction.
yhat=filter((nh-dh),nh,y);
for i=1:num_inputs,
  yhat = yhat + filter(conv(dh,ng{i}),conv(nh,dg{i}),u(i,:));
end
