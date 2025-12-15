function plotindex(tr,goal,name,epoch)
%PLOTPERF Plot prediction model performance index.
%
%	Syntax
%
%	  plotindex(tr,goal,name,epoch)
%
%	Description
%
%	  PLOTINDEX(TR,GOAL,NAME,EPOCH) takes these inputs,
%	    TR - Training record returned by train.
%	    GOAL - Performance goal, default = NaN.
%	    NAME - Training function name, default = ''.
%	    EPOCH - Number of epochs, default = length of training record.
%	  and plots the training performance index and the performance
%	  goal.
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
%	  to a error goal of 0.01. 
%	  
%	    pmod.estimParam.epochs = 50;
%	    pmod.estimParam.goal = 0.01;
%	    [pmod,trec,stat] = estimate(pmod,y,u);
%
%	  During the estimation, PLOTINDEX was called to display the estimation
%	  record.  You can also call PLOTINDEX directly with the final
%	  estimation record TREC, as shown below.
%
%	    plotindex(trec)
%

% Mark Beale, 1-31-92
% Revised from network training to pmod training 7-1-00
% Yong Hu, Martin Hagan, 9-15-00
% $Revision: 1.0 $ $Date: 21-Sep-2000 14:37:36 $

if nargin < 2, goal = NaN; end
if nargin < 3, name = ''; end
if nargin < 4, epoch = length(tr.epoch)-1; end

% Get figure if it exists, or create new figure
me = ['Training with ' name];
fig = chkfig(me);
if length(get(fig,'children')) == 0, fig = 0; end
if (exist('fig')&&(fig~=0))
  figure(fig);
else
  fig = figure;
  clf reset
  set(fig,'name',me,'numbertitle','off')
end

ind = 1:(epoch+1);
printGoal = isfinite(goal);
plotGoal = isfinite(goal) & (goal > 0);

semilogy(tr.epoch(ind),tr.index(ind),'linewidth',2);
hold off

tstring = sprintf('Performance is %g',tr.index(epoch+1));
if printGoal
  tstring = [tstring ', ' sprintf('Goal is %g',goal)];
end
title(tstring)

if epoch == 0
  xlabel('Zero Epochs')
elseif epoch == 1
  xlabel('One Epoch')
else
  xlabel([num2str(epoch) ' Epochs'])
end

ystring = 'Performance Index';
ylabel(ystring)
%if length(name)
%  set(gcf,'name',['Training with ' name],'numbertitle','off')
%end
if epoch > 0
  set(gca,'xlim',[0 epoch])
end
drawnow
