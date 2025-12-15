function pmod=subsasgn(pmod,subscripts,v)
%SUBSASGN  Assign fields of a prediction model

% $Date: 08-05-00 $
% $Revision: 1.1 $

% Assume no error
err = '';

% First subscript
[subscripts,field,type,moresubs] = nextsubs(subscripts);

switch type

case '.'
  field = matchstring(field,fieldnames(pmod));
  
  switch(field)
    
    case 'type'
      [typ,err] = nsubsasn(pmod.type,subscripts,v);
      if length(err),error(err),end
      [pmod,err] = setType(pmod,typ);
    case 'a'
      if (~moresubs & ~iscell(v))
         error('Attempt to assign field of non-cell array');
      end;
      [pmod.a,err] = nsubsasn(pmod.a,subscripts,v);
      if length(err),error(err),end
    case 'b'
      if (~moresubs & ~iscell(v))
         error('Attempt to assign field of non-cell array');
      end;
      [pmod.b,err] = nsubsasn(pmod.b,subscripts,v);
      if length(err),error(err),end
    case 'c'
      if (~moresubs & ~iscell(v))
         error('Attempt to assign field of non-cell array');
      end;
      [pmod.c,err] = nsubsasn(pmod.c,subscripts,v);
      if length(err),error(err),end
    case 'd'
      if (~moresubs & ~iscell(v))
         error('Attempt to assign field of non-cell array');
      end;
      [pmod.d,err] = nsubsasn(pmod.d,subscripts,v);
      if length(err),error(err),end
    case 'f'
      if (~moresubs & ~iscell(v))
         error('Attempt to assign field of non-cell array');
      end;
      [pmod.f,err] = nsubsasn(pmod.f,subscripts,v);
      if length(err),error(err),end
    case 'diff'
      [pmod.diff,err] = nsubsasn(pmod.diff,subscripts,v);
      if length(err),error(err),end
    case 'period'
      [pmod.period,err] = nsubsasn(pmod.period,subscripts,v);
      if length(err),error(err),end
    case 'delay'
      [pmod.delay,err] = nsubsasn(pmod.delay,subscripts,v);
      if length(err),error(err),end
    case 'estimFcn'
      [estimFcn,err] = nsubsasn(pmod.estimFcn,subscripts,v);
      if length(err),error(err),end
      [pmod,err] = setEstimFcn(pmod,estimFcn);
    case 'indexFcn'
      [indexFcn,err] = nsubsasn(pmod.indexFcn,subscripts,v);
      if length(err),error(err),end
      [pmod,err] = setIndexFcn(pmod,indexFcn);  
    case 'initFcn'
      [initFcn,err] = nsubsasn(pmod.initFcn,subscripts,v);
      if length(err),error(err),end
      [pmod,err] = setInitFcn(pmod,initFcn);  

    % Estimation Parameters
    case 'estimParam'
      %[pmod.estimParam,err] = nsubsasn(pmod.estimParam,subscripts,v);
      %if length(err),error(err),end
      [subscripts,field,type,moresubs] = nextsubs(subscripts);
      if strcmp(type,'{}'),error('Cell contents assignment to a non-cell array object.'),end
      if strcmp(type,'()'),,error('Array contents assignment to a non-array object.'),end
      field = matchstring(field,fieldnames(pmod.estimParam));
      
      switch(field)
       
        case 'epochs'
          [pmod.estimParam.epochs,err] = nsubsasn(pmod.estimParam.epochs,subscripts,v);
          if length(err),error(err),end
        case 'goal'
          [pmod.estimParam.goal,err] = nsubsasn(pmod.estimParam.goal,subscripts,v);
          if length(err),error(err),end
        case 'min_grad'
          [pmod.estimParam.min_grad,err] = nsubsasn(pmod.estimParam.min_grad,subscripts,v);
          if length(err),error(err),end
        case 'mu'
          [pmod.estimParam.mu,err] = nsubsasn(pmod.estimParam.mu,subscripts,v);
          if length(err),error(err),end
        case 'mu_dec'
          [pmod.estimParam.mu_dec,err] = nsubsasn(pmod.estimParam.mu_dec,subscripts,v);
          if length(err),error(err),end
        case 'mu_inc'
          [pmod.estimParam.mu_inc,err] = nsubsasn(pmod.estimParam.mu_inc,subscripts,v);
        if length(err),error(err),end
        case 'mu_max'
          [pmod.estimParam.mu_max,err] = nsubsasn(pmod.estimParam.mu_max,subscripts,v);
          if length(err),error(err),end
        case 'show'
          [pmod.estimParam.show,err] = nsubsasn(pmod.estimParam.show,subscripts,v);
          if length(err),error(err),end
        case 'time'
          [pmod.estimParam.time,err] = nsubsasn(pmod.estimParam.time,subscripts,v);
          if length(err),error(err),end
        case 'delta'
          [pmod.estimParam.delta,err] = nsubsasn(pmod.estimParam.delta,subscripts,v);
          if length(err),error(err),end
        otherwise
          error('Reference to non-existent field.') 
      end
       
    case 'upreproc'
      [pmod.upreproc,err] = nsubsasn(pmod.upreproc,subscripts,v);
      if length(err),error(err),end
    case 'ypreproc'
      [pmod.ypreproc,err] = nsubsasn(pmod.ypreproc,subscripts,v);
      if length(err),error(err),end
    case 'ypostproc'
      [pmod.ypostproc,err] = nsubsasn(pmod.ypostproc,subscripts,v);
      if length(err),error(err),end
      
    otherwise, error('Reference to non-existent field.');
    end
    
case '{}',error('Cell contents assignment to a non-cell array object.')
case '()',error('Array contents assignment to a non-array object.')
end

% Error message
if length(err), error(err), end






% ===========================================================
%                          UTILITY FUNCTIONS
% ===========================================================

function [subscripts,subs,type,moresubs]=nextsubs(subscripts)
% NEXTSUBS get subscript data from a subscript array.

subs = subscripts(1).subs;
type = subscripts(1).type;
subscripts(1) = [];
moresubs = length(subscripts) > 0;

% ===========================================================

function field = matchstring(field,strings)
% MATCHFIELD replaces FIELD with any field belonging to STRUCTURE
% that is the same when case is ignored.

field2 = lower(field);

for i=1:length(strings)
  if strcmp(field2,lower(strings{i}))
    field = strings{i};
    break;
  end
end

% ===========================================================


function [o,err]=nsubsasn(o,subscripts,v)
%NSUBSASN General purpose subscript assignment.

% Assume no error
err = '';

% Null case
if length(subscripts) == 0
  o = v;
  return
end

type = subscripts(1).type;
subs = subscripts(1).subs;
subscripts(1) = [];
o2 = [];

  % Paretheses
  switch type
  
  case '()'
    eval('o2=o(subs{:});','err=lasterr');
	if length(err), return, end
    [v,err] = nsubsasn(o2,subscripts,v);
	if length(err), return, end
	
    eval('o(subs{:})=v;','err=lasterr;')
	
  % Curly bracket
  case '{}'
    eval('o2=o{subs{:}};','err=lasterr');
	if length(err), return, end
    [v,err] = nsubsasn(o2,subscripts,v);
	if length(err), return, end
	
    eval('o{subs{:}}=v;','err=lasterr;')
    
  % Dot
  case '.'

    % Match field name regardless of case
    if isa(o,'struct') | isa(o,'network')
	  found = 0;
      f = fieldnames(o);
	  for i=1:length(f)
	    if strcmp(subs,lower(f{i}))
	      subs = f{i};
		  found = 1;
		  break;
	    end
	  end
	  if (~found)
	    eval(['o.' subs '=v;'],'err=lasterr');
		return
	  end
	else
	  err = 'Attempt to reference field of non-structure array.';
	  return
	end

    eval(['o2=o.' subs ';'],'err=lasterr');
	if length(err), return, end
    [v,err] = nsubsasn(o2,subscripts,v);
	if length(err), return, end

	eval(['o.' subs '=v;'],'err=lasterr;')
    
  end
  
% ===========================================================

function [pmod,err] = setEstimFcn(pmod,estimFcn)

% Check value
err = '';
if ~ischar(estimFcn)
  err = sprintf('"estimFcn" must be ''''.');
  return
end
if length(estimFcn) & ~exist(estimFcn)
  err = sprintf('"estimFcn" cannot be set to non-existing estimation function "%s".',estimFcn);
  return
end

% Change function
pmod.estimFcn = estimFcn;

% Default parameters
if length(estimFcn)
  pmod.estimParam = feval(estimFcn,'pdefaults');
else
  pmod.estimParam = [];
end

% ===========================================================

function [pmod,err] = setIndexFcn(pmod,indexFcn)

% Check value
err = '';
if ~ischar(indexFcn)
  err = sprintf('"indexFcn" must be '''' .');
  return
end
if length(indexFcn) & ~exist(indexFcn)
  err = sprintf('"indexFcn" cannot be set to non-existing index function "%s".',indexFcn);
  return
end

%indexFcn must begin with 'pmod',
% and should not be 'pmodaic','pmodbic','pmodsim' and 'pmod'
if (~strncmp('pmod',indexFcn,4) |...
   strcmp(indexFcn,'pmodaic') |...
   strcmp(indexFcn,'pmodbic') |...
   strcmp(indexFcn,'pmodsim') |...
   strcmp(indexFcn,'pmod'))
   err = sprintf('"%s" is not a valid index function.',indexFcn);
   return
end

% Change function
pmod.indexFcn = indexFcn;

% ===========================================================

function [pmod,err] = setInitFcn(pmod,initFcn)

% Check value
err = '';
if ~ischar(initFcn)
  err = sprintf('"initFcn" must be '''' .');
  return
end
if length(initFcn) & ~exist(initFcn)
  err = sprintf('"initFcn" cannot be set to non-existing init function "%s".',initFcn);
  return
end

%initFcn must begin with 'init', but 'init' is not proper.
if ((~strncmp('init',initFcn,4)) | (strcmp('init',initFcn)))
   err = sprintf('"%s" is not a valid init function.',initFcn);
   return
end

% Change function
pmod.initFcn = initFcn;

% ===========================================================

function [pmod,err] = setType(pmod,typ)

% Check value
err = '';
if ~ischar(typ)
  err = sprintf('"type" must be '''' .');
  return
end
if length(typ) & ~exist(['pred' typ])
  err = sprintf('"type" cannot be set to non-existing type "%s".',typ);
  return
end

% Change function
pmod.type = typ;

% ===========================================================







