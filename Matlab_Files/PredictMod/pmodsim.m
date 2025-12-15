function y = pmodsim(pmod,e,u)
%PMODSIM Simulate a prediction model.
%
%	Syntax
%
%	  [Y] = PMODSIM(PMOD,E,U)
%
%	Description
%
%	  PMODSIM simulates the prediction model PMOD.
%
%	  PMODSIM(PMOD,E,U) takes,
%	    PMOD - Prediction model.
%	    E    - White noise.
%	    U    - Prediction model inputs.
%	  and returns,
%	    Y    - Prediction model output.
%
%
%	Examples
%
%	  Here are a few points from sample e and u sequences:
%
%	    e = [ 0.9501    0.2311    0.6068    0.4860    0.8913    0.7621];
%	    u = [-0.4326   -1.6656    0.1253    0.2877   -1.1465    1.1909];
%
%	  Here NEWBJTF is used to create a Box and Jenkins Transfer Function
%	  model with first order polynomials.
%
%	    pmod = newbjtf(1,1,1,1);
%
%	  The parameters of the model are all set to random values by default. 
%	  Now we can simulate this random prediction model: 
%
%	    y = pmodsim(pmod,e,u);
%	    ind = 1:length(y);
%	    plot(ind,u,'o',ind,y,'x')
%	    
%	Algorithm
%
%	  PMODSIM simulates the following system model.
%
%     y(t) = sumi{Gi ui(t)} + H e(t)
%
%	See also PREDICT, ESTIMATE

% Yong Hu, Martin Hagan, 9-15-00
% $Revision: 1.0 $ $Date: 21-Sep-2000 14:37:36 $


%preprocessing
uflag = nargin>2;
if uflag,
  u = makerow(u);
end
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


if strcmp(pmod.type,'regr') %regression model
   
   [ru,cu] = size(u);
   for i=1:ru
      if isempty(pmod.delay)
         idel = 0;
      else
         idel = pmod.delay(i);
      end
      udelay(i,:) = [zeros(1,idel) u(i,1:cu-idel)];
   end
   u1 = [ones(1,length(e));udelay];
   y = pmod.b{1}*u1 + e;

else %bjtf, arma, armax, arx model

  % Expand the parameter vectors into g and h form
  [ng,dg,nh,dh] = getGH(pmod);
  num_inputs = length(ng);
  diff = pmod.diff;
  period = pmod.period;

  % Add the differencing terms into H
  for i=1:diff(1),
  dh = conv(dh,[1 -1]);
  end
  lp = length(period);
  for i=1:lp,
    per = period(i);
    ddh = [1 zeros(1,per-1) -1];
    for j=1:diff(i+1),
      dh = conv(dh,ddh);
    end
  end

  % Simulate the prediction model
  y=filter(nh,dh,e);
  for i=1:num_inputs,
    y = y + filter(ng{i},dg{i},u(i,:));
  end

end


%post processing
ypostproc = pmod.ypostproc;
pc = size(ypostproc,2); %only one output is possible
if (pc & size(ypostproc,1)~=1)
   error('ypostproc should have only one row. ');
end
if pc~=0
   for i=1:pc,
      y = feval(ypostproc{i},y);
   end
end





