function [pmod,trec,stat] = estimate(pmod,y,u)
%ESTIMATE Estimate the parameters for a prediction model.
%
%	Syntax
%
%	  [PMOD,TREC,STAT] = ESTIMATE(PMOD,Y,U)
%
%	Description
%
%	  ESTIMATE estimates the parameters in a prediction model
%	  PMOD according to PMOD.estimFcn and PMOD.estimParam.
%
%	  ESTIMATE(PMOD,Y,U) takes,
%	    PMOD - Prediction model.
%	    Y    - Prediction model desired outputs.  Y may or may not be a
%	           structure.  If Y is a structure, then Y.Y is the prediction 
%	           model desired outputs, and Y.M is a vector
%	           containing the weighting factors for each error.  
%	    U    - Prediction model inputs.
%	  and returns,
%	    PMOD - New prediction model.
%	    TREC - Training record (index).
%	    STAT - Statistics for final model.
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
%	  ESTIMATE calls the function indicated by PMOD.estimFcn, using the
%	  estimation parameters indicated by PMOD.estimParam.
%
%	  Typically one epoch of estimation is defined as a single presentation
%	  of the entire input sequence to the model.  The model parameters are 
%	  then updated.
%
%	  Estimation continues until a maximum number of epochs occurs, the
%	  performance goal is met, or any other stopping condition of the
%	  function PMOD.estimFcn occurs.
%
%	See also PMODSIM, PREDICT

% Yong Hu, Martin Hagan, 9-15-00
% $Revision: 1.0 $ $Date: 21-Sep-2000 14:37:36 $

% Make sure that the inputs are in row format
uflag = nargin>2;

if uflag,
  u = makerow(u);
end
[ystru,y,m] = sepym(y);
y = makerow(y);

% Preprocess the sequences
upreproc = pmod.upreproc;
pr = size(upreproc,1);
if (uflag & pr~=0)
   if (pr~=1 & pr~=size(u,1))
      error('rows of upreproc should either equals 1 or the number of inputs. ');
   end
   if pr==1
      pc = upreproc(pr,:);
      for i=1:pc
         u = feval(upreproc{pr,i},u);
      end
   else
      for i=1:pr
         pc = length(upreproc(i,:));
         for j=1:pc
            if ~isempty(upreproc{i,j})
               u(i,:) = feval(upreproc{i,j},u(i,:));
            end
         end
      end
   end
end

ypreproc = pmod.ypreproc;
pc = size(ypreproc,2); %only one output is possible
if (pc & size(ypreproc,1)~=1)
   error('ypreproc should have only one row. ');
end
if pc~=0,
   for i=1:pc,
      y = feval(ypreproc{i},y);
   end
end

% Difference the sequences
period = [1 pmod.period];
diff = pmod.diff;
for i=1:length(diff),
  d = diff(i);
  if (d~=0),
    if uflag,
      u = sdiff(u,d,period(i));
    end
    y = sdiff(y,d,period(i));
  end
end

%check to see if y,u are zero mean
if abs(mean(y))>(2*std(y))
   warning('The desired output may not be a zero mean sequence.');
end
if uflag & any(abs(mean(u,2))>(2*std(u,0,2)))
   warning('Input may not be zero mean sequences.');
end

% Call the appropriate estimation function
ystru.y = y;
ystru.m(1:length(m)-length(y)) = []; 
if uflag,
  [pmod,trec,stat] = feval(pmod.estimFcn,pmod,ystru,u);
else
  [pmod,trec,stat] = feval(pmod.estimFcn,pmod,ystru);
end


