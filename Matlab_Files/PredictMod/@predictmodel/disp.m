function disp(pmod)
%DISP Display a prediction model's properties.
%
%	Syntax
%
%	  disp(pmod)
%
%	Description
%
%	  DISP(pmod) displays a model's properties.
%
%	Examples
%
%	  Here a predict model is created and displayed.
%
%	    pmod = predictmodel;
%	    disp(pmod)
%
%	See also DISPLAY

% $07-21-00
% $Revision: 1.0 $

isLoose = strcmp(get(0,'FormatSpacing'),'loose');

if (isLoose), fprintf('\n'), end
fprintf('    Prediction Model object:\n');
if (isLoose), fprintf('\n'), end
fprintf('              type: %s\n',pmod.type);
if (isLoose), fprintf('\n'), end
fprintf('  model parameters:\n');
if (isLoose), fprintf('\n'), end

fprintf('                       a: %s\n',boolstr(pmod.a));
fprintf('                       b: %s\n',boolstr(pmod.b));
fprintf('                       c: %s\n',boolstr(pmod.c));
fprintf('                       d: %s\n',boolstr(pmod.d));
fprintf('                       f: %s\n',boolstr(pmod.f));
fprintf('                    diff: %s\n',boolstr(pmod.diff));
fprintf('                  period: %s\n',boolstr(pmod.period));
fprintf('                   delay: %s\n',boolstr(pmod.delay));

if (isLoose), fprintf('\n'), end
fprintf('         functions:\n');
if (isLoose), fprintf('\n'), end
fprintf('                estimFcn: %s\n',pmod.estimFcn);
fprintf('                indexFcn: %s\n',pmod.indexFcn);
fprintf('                 initFcn: %s\n',pmod.initFcn);


if (isLoose), fprintf('\n'), end
fprintf('  estim parameters:\n');
if (isLoose), fprintf('\n'), end
fprintf('  pmod.estimParam.epochs: %g\n',pmod.estimParam.epochs);
fprintf('    pmod.estimParam.goal: %g\n',pmod.estimParam.goal);
fprintf('pmod.estimParam.min_grad: %g\n',pmod.estimParam.min_grad);
fprintf('      pmod.estimParam.mu: %g\n',pmod.estimParam.mu);
fprintf('  pmod.estimParam.mu_dec: %g\n',pmod.estimParam.mu_dec);
fprintf('  pmod.estimParam.mu_inc: %g\n',pmod.estimParam.mu_inc);
fprintf('  pmod.estimParam.mu_max: %g\n',pmod.estimParam.mu_max);
fprintf('    pmod.estimParam.show: %g\n',pmod.estimParam.show);
fprintf('    pmod.estimParam.time: %g\n',pmod.estimParam.time);
fprintf('   pmod.estimParam.delta: %g\n',pmod.estimParam.delta);

if (isLoose), fprintf('\n'), end
fprintf('pre-post processor:\n');
if (isLoose), fprintf('\n'), end
fprintf('                upreproc: %s\n',dispstr(pmod.upreproc));
fprintf('                ypreproc: %s\n',dispstr(pmod.ypreproc));
fprintf('               ypostproc: %s\n',dispstr(pmod.ypostproc));

if (isLoose), fprintf('\n'), end



% ===========================================================

function s=boolstr(b)
%BOOLSTR displays a row cell array,or numeric array in a string

%cell array
if iscell(b)
   if isempty(b)
      s = sprintf('{}');
      return;
   end
   if size(b,1)~=1
      error('cell dimension should be 1 by n');
      return;
   end
   if prod(size(b)) > 12
      s = sprintf('[%gx%g boolean]',size(b,1),size(b,2));
   else
      s = '{';
      for i=1:size(b,2)
      	if i~=1
         s = [s sprintf(' ')];
         end
         for j=1:length(b{i})
            if j==1 
               s = [s sprintf('[%g',b{i}(j))];
         	elseif j==length(b{i})
            	s = [s sprintf(' %g] ',b{i}(j))];
         	else
          	   s = [s sprintf(' %g',b{i}(j))];
            end  
            if length(b{i})==1
               s = [s sprintf('] ')];
            end
         end
      end
      if length(s)~=1,
         s(length(s)) = [];
      end
      s = [s sprintf('}')];
   end
  
 %numerical array 
else
   if isempty(b)
      s = sprintf('[]');
      return;
   end
   if size(b,1)~=1
      error('array dimension should be 1 by n');
      return;
   end
   if prod(size(b)) > 12
      s = sprintf('[%gx%g boolean]',size(b,1),size(b,2));
   else
      s = '[';
      for i=1:size(b,2)
         if i~=1
            s = [s sprintf(' ')];
         end
         s = [s sprintf('%g',b(i))];
      end
      s = [s sprintf(']')];
   end
end

  
% ===========================================================

function s=dispstr(b)
%DISPSTR displays a cell which contains only strings

if isempty(b)
   s = '{}';
   return;
end

s = '{';
%for i=1:length(b)
%   if i==1
%      s = [s sprintf('''%s''',b{i})];
%   else
%      s = [s sprintf(' ''%s''',b{i})];
%   end
%end
for i=1:size(b,1)
   if i~=1
      s = [s ' ; '];
   end
   for j=1:size(b,2)
      if j==1
         s = [s sprintf('''%s''',b{i,j})];
      else
         if isempty(b{i,j})
            s = [s sprintf(',[]')];
         else
            s = [s sprintf(',''%s''',b{i,j})];
         end
      end
   end
end

s = [s '}'];
     






            
         
         




