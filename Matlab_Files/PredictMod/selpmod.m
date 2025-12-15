function [estpmod] = selpmod(filename,y,u)
%SELPMOD Select the best prediction model based on AIC and BIC criteria.
%
%	Syntax
%
%	  [estpmod] = selpmod(filename,y,u)
%	  [estpmod] = selpmod(string,y,u)
%	  [estpmod] = selpmod(cellarray,y,u)
%
%	Description
%
%	  SELPMOD select the best prediction model (model parameters defined
%	  in 'filename') from data y, u based on aic and bic criteria.  
%
%	  SELPMOD(FILENAME,Y,U) takes,
%	    FILENAME - File name, cell array or string which specifies the estimation parameters.
%	    Y        - Prediction model desired outputs.
%	    U        - Prediction model inputs, default = [] (only for arma model).
%	  and returns,
%	    ESTPMOD  - Estimated prediction model, which has the following structure:
%	      ESTPMOD.'TYPE'.AIC     - AIC array, along with the corresponding
%	                              order parameters, such as na,nb,nc,nd.
%	      ESTPMOD.'TYPE'.BIC     - BIC array, along with the corresponding
%	                              order parameters, such as na,nb,nc,nd.
%	      ESTPMOD.'TYPE'.AICSTAT.SIGMA - AIC model sum squared error.
%	      ESTPMOD.'TYPE'.AICSTAT.STDX  - AIC model parameter standard deviation.
%	      ESTPMOD.'TYPE'.BICSTAT.SIGMA - BIC model sum squared error.
%	      ESTPMOD.'TYPE'.BICSTAT.STDX  - BIC model parameter standard deviation.
%	      ESTPMOD.'TYPE'.AICMOD - The best prediction model selected from AIC criteria.
%	      ESTPMOD.'TYPE'.BICMOD - The best prediction model selected from BIC criteria.
%	      'TYPE' can be 'bjtf','arx','arma','armax','regr'.	      
%
%	  FILENAME is the file that specify parameter ranges (orders of na,nb,nc,nd,nf,
%	  ranges of delay,diff,per, etc.).  It should meet the following specifications:
%	    (1) each line can only specify the order parameters for one model type.
%	    (2) each line begins with model type, quoted with single quotation.
%	    (3) each line ends with a full stop '.'.
%	    (4) parameters can be specified as follows:
%	          na=1-4   -> na = [1 2 3 4]
%	          na=1~4   -> na = [1 2 3 4]
%	          na=1,2   -> na = [1 2]
%	          na=1-3,5 -> na = [1 2 3 5]
%	    (4) delimiter can be '\t', ' ' , ',' , ';'.
%	    (5) delimiters which are ajacent to each other are treated as one delimiter.
%	    (6) lines after '%' are treated as comments.
%	    (7) case is ignored.
%	  
%	  SELPMOD(STRING,...) takes string as the first arguement.  The string has the 
%	  same specification as filename does.
%
%	  SELPMOD(CELLARRAY,...) takes cell array as the first argument.  Each element of 
%	  the cell is a string, which meets the same specification as filename does.
%
%
%	Examples
%
%	  Here is the real model parameters and y,u data generated from
%	  the real model.
%
%	    nb = 1; 
%	    na = 2;
%	    delay = 1;
%	    b = {[2 3]};
%	    a = {[-1 0.25]};
%	    u=randn(1,2000);
%	    e=randn(1,2000)*.5;
%	    pmodr = newarx(na,nb,delay);
%	    pmodr.a = a;
%	    pmodr.b = b;
%	    y = pmodsim(pmodr,e,u);
%
%	  The models to be estimated can be specified as follows, 
%
%	   SPEC{1} = '''arx'' ,na=1-3;nb=1~2,delay=0,1,2,diff=1-2,3.';
%	   SPEC{2} = '''armax'',na=1,2,nb=1,nc=0-1,nd=1,delay=0-1.';
%
%	  Here the best prediction models are estimated based on AIC 
%	  and BIC,
%	
%	    estpmod = selpmod(SPEC,y,u);
%	    Mreal = getmX(pmodr)
%	    Marxaic = getmX(estpmod.arx.aicmod)
%	    Marxbic = getmX(estpmod.arx.bicmod)
%	    Marmaxaic = getmX(estpmod.armax.aicmod)
%	    Marmaxbic = getmX(estpmod.armax.bicmod)
%
%	See also PREDICT, PMODSIM, ESTIMATE.

% Yong Hu, Martin Hagan, 9-15-00
% $Revision: 1.0 $ $Date: 21-Sep-2000 14:37:36 $

if nargin < 2
   error('Not enough input arguments')
end
if nargin < 3
   u = [];
end

% get parameters from file
if iscell(filename)
   file = filename;
elseif exist(filename)
   file = textread(filename,'%s',...
      'commentstyle','matlab',...
      'delimiter','\n',...
      'whitespace','\n');
elseif ischar(filename)
   file{1} = filename;
else
   error('First input should either be a filename or a string.');
end

file = deblank(file);
file = lower(file);
len = length(file);
i = 1;
while i<=len
   if isempty(findstr(file{i},'.'))
      file(i) = [];
      len = len-1;
   else
      i = i+1;
   end
end


%Hmsg = msgbox('Selecting Model Now, Please Wait!', 'Reminder','warn');
%set(findobj(Hmsg,'Tag','OKButton'),'Visible','OFF');
   
for i=1:length(file)
   temp = file{i};
   if (temp(length(temp))~='.')
      error('each line of the file should end with "."');
   end
   ind = findstr(temp,'''');
   if (size(ind,1)~=1 & size(ind,2)~=2)
      error('each line of the file should begin by specifying model type');
   end
   type = temp(ind(1)+1:ind(2)-1);
   temp(1:ind(2)) = [];
   temp = clrdelim(temp,1);
      
   %find na,nb,nc,nd,nf,diff,per,delay
   [na,temp] = findspec(temp,'na');
   [nb,temp] = findspec(temp,'nb');
   [nc,temp] = findspec(temp,'nc');
   [nd,temp] = findspec(temp,'nd');
   [nf,temp] = findspec(temp,'nf');
   [diff,temp] = findspec(temp,'diff');
   [per,temp] = findspec(temp,'per');
   [delay,temp] = findspec(temp,'delay');
   
   %set default na,nb,nc,nd,nf,diff,per,delay
   if isempty(na), na = [0 1]; end
   if isempty(nb), nb = [0 1]; end
   if isempty(nc), nc = [0 1]; end
   if isempty(nd), nd = [0 1]; end
   if isempty(nf), nf = [0 1]; end
   if isempty(diff), diff = [0]; end
   %if isempty(per), per = []; end
   if isempty(delay), delay = [0 1]; end
   
   %automatic model selection
   
   switch type
     
   case 'arx'
      lna = 1;
      lnb = size(u,1);
      lDel = lnb;
      A = na;
      for i=1:lnb
         A = gcombvec(A,nb);
      end
      for i=1:lDel
         A = gcombvec(A,delay);
      end
      TIter = size(A,2);
   case 'arma'
      %selarma(y,u,maxOrd,maxDel,lnc,maxDiff,maxPer)
      lnc = length(per)+1;
      lnd = lnc;
      lDiff = lnc;
      lPer = length(per);
      A = nc;
      for i=1:lnc-1
         A = gcombvec(A,nc);
      end
      for i=1:lnd
         A = gcombvec(A,nd);
      end
      for i=1:lDiff
         A = gcombvec(A,diff);
      end
      for i=1:lPer
         A = gcombvec(A,per);
      end
   case 'armax'
      lna = 1;
      lnb = size(u,1);
      lnc = 1;
      lDel = lnb;
      A = na;
      for i=1:lnb
         A = gcombvec(A,nb);
      end
      A = gcombvec(A,nc);
      for i=1:lDel
         A = gcombvec(A,delay);
      end
   case 'bjtf'
      lnb = size(u,1);
      lnc = length(per)+1;
      lnd = lnc;
      lnf = lnb;
      lDel = lnb;
      lDiff = lnc;
      lPer = length(per);
      A = nb;
      for i=1:lnb-1
         A = gcombvec(A,nb);
      end
      for i=1:lnc
         A = gcombvec(A,nc);
      end
      for i=1:lnd
         A = gcombvec(A,nd);
      end
      for i=1:lnf
         A = gcombvec(A,nf);
      end
      for i=1:lDel
         A = gcombvec(A,delay);
      end
      for i=1:lDiff
         A = gcombvec(A,diff);
      end
      for i=1:lPer
         A = gcombvec(A,per);
      end
   case 'regr'
      lnb = size(u,1)+1;
      lDel = lnb-1;
      A = delay;
      for i=1:lDel-1
         A = gcombvec(A,delay);
      end
   otherwise
      error('Unrecognized model type.') 
   end
   
   TIter = size(A,2);

   %compute aic, aicmod, aicstat, bic, bicstat, bicmod
   aic = [];
   bic = [];
   aicstat = [];
   bicstat = [];
   minaic = inf;
   minbic = inf;
   iter  = 0;
   for i=A
      temp = i';
      
      %get model inputs
      switch type
      case 'arx'
         na = temp(1);
         temp(1) = [];
         nb = temp(1:lnb);
         temp(1:lnb) = [];
         delay = temp;
         pmodt = newarx(na,nb,delay);
      case 'arma'
         nc = temp(1:lnc);
         temp(1:lnc) = [];
         nd = temp(1:lnd);
         temp(1:lnd) = [];
         diff = temp(1:lDiff);
         temp(1:lDiff) = [];
         per = temp;
         pmodt = newarma(nc,nd,diff,per);
      case 'armax'
         na = temp(lna);
         temp(1) = [];
         nb = temp(1:lnb);
         temp(1:lnb) = [];
         nc = temp(lnc);
         temp(1) = [];
         delay = temp;
         pmodt = newarmax(na,nb,nc,delay);
      case 'bjtf'
         nb = temp(1:lnb);
         temp(1:lnb) = [];
         nc = temp(1:lnc);
         temp(1:lnc) = [];
         nd = temp(1:lnd);
         temp(1:lnd) = [];
         nf = temp(1:lnf);
         temp(1:lnf) = [];
         delay = temp(1:lnb);
         temp(1:lnb) = [];
         diff = temp(1:lnc);
         temp(1:lnc) = [];
         per = temp;
         pmodt = newbjtf(nb,nc,nd,nf,delay,diff,per);
      case 'regr'
         delay = temp;
         nb = lnb-1;
         pmodt = newregr(nb,delay);
      end
      
      pmodt = init(pmodt);
      pmodt.estimParam.show = NaN;
      pmodt.estimParam.epochs = 50;
      pmodt.estimParam.goal = 0.01;
      if isempty(u)
         [pmodt,trec,stat] = estimate(pmodt,y);
         aictmp = pmodaic(pmodt,y);
         bictmp = pmodbic(pmodt,y);
      else
         [pmodt,trec,stat] = estimate(pmodt,y,u);
         aictmp = pmodaic(pmodt,y,u);
         bictmp = pmodbic(pmodt,y,u);
      end
      aic = [aic aictmp];
      if minaic > aictmp;
         minaic = aictmp;
         aicmod = pmodt;
         aicstat = stat;
      end
      bic = [bic bictmp];
      if minbic > bictmp;
         minbic = bictmp;
         bicmod = pmodt;
         bicstat = stat;
      end
      
      %display iteration
      iter = iter +1;
      if iter ==1
         str = sprintf('Selecting the best %s prediction model',upper(type));
         disp(str);
      end
      str = sprintf('%s: Combination %g out of %g total.  aic = %g, bic = %g',...
         type,iter,TIter,aictmp,bictmp);
      disp(str);
   end
   aic = [aic;A];
   bic = [bic;A];
   
   eval(sprintf('estpmod.%s.aic = aic;',type));
   eval(sprintf('estpmod.%s.bic = bic;',type));
   eval(sprintf('estpmod.%s.aicstat = aicstat;',type));
   eval(sprintf('estpmod.%s.bicstat = bicstat;',type));
   eval(sprintf('estpmod.%s.aicmod = aicmod;',type));
   eval(sprintf('estpmod.%s.bicmod = bicmod;',type));
end

%delete(Hmsg);



% ===========================================================

function [para,strs] = findspec(strs,str)
%FINDSPEC Find the specified parameter value from strs.
%	  name of the parameter is specified in str.

para = [];
temp = strs;
ind = findstr(temp,str);
if ind
   temp(ind:ind+length(str)-1) = [];
   temp = clrdelim(temp,ind);
   if temp(ind) ~= '='
      error('the proper format should be na=1...');
   end
   temp(ind) = [];
   temp = clrdelim(temp,ind);
   while (temp(ind)>='0' & temp(ind)<='9')
      fst = str2num(temp(ind));
      temp(ind) = [];
      temp = clrdelim(temp,ind);
      if (temp(ind) == '-' | temp(ind) == '~')
         temp(ind) = [];
         temp = clrdelim(temp,ind);
         if (temp(ind)>'0' & temp(ind)<'9')
            snd = str2num(temp(ind));
            para = [para fst:snd];
            temp(ind) = [];
            temp = clrdelim(temp,ind);
         else
            error('unrecognized character');
         end
      elseif ((temp(ind)>='0' & temp(ind)<='9') |...
            (temp(ind)=='.') | (temp(ind)=='n') |...
            (temp(ind)=='d') | (temp(ind)=='p'))
         para = [para fst];
         temp = clrdelim(temp,ind);
      else
         error('unrecognized character');
      end
   end
   %if let per=[], eliminate '[]'.
   if (length(temp)>(ind+1))
      if (temp(ind)=='[' | temp(ind+1)==']')
         temp(ind:ind+1)=[];
      end
   end
   temp = clrdelim(temp,ind);
end
strs = temp;

% ===========================================================

function str =  clrdelim(str,i)
%clear string delimiter starting from position i
while (str(i)==',' | str(i)=='	' |...
      str(i)==' ' | str(i)==';')
   str(i) = [];
end

% ===========================================================

