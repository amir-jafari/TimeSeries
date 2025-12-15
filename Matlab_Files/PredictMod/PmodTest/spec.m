% format of spec.m file :
%	  (1) each line can only specify the order parameters for one model type.
%	  (2) each line begins with model type, quoted with single quotation.
%	  (3) each line ends with a full stop '.'.
%	  (4) parameters can be specified as follows:
%	        na=1-4   -> na = [1 2 3 4]
%	        na=1~4   -> na = [1 2 3 4]
%	        na=1,2   -> na = [1 2]
%	        na=1-3,4 -> na = [1 2 3 4]
%	  (4) delimiter can be '\t', ' ' , ',' , ';'.
%	  (5) delimiters which are ajacent to each other are treated as one delimiter.
%	  (6) lines after '%' are treated as comments.
%	  (7) case is ignored.
%


%GENERAL TEST,Used with testselpmod
%'arx' , na = 1 - 3 ; nb = 1 ~ 2 , delay = 0 , 1 , 2 , diff  =  1-2	,	3	.
%'arma',nc=1,2,nd=0,2,per =[].
%'armax',na=1,2,nb=1,nc=1-2,nd=1,delay=0-1.
%'bjtf',nb=1,nc=0,1,2,nd=1-2,nf=1-2,delay=1,diff=0.
%'regr',nb=3,delay=0-1.

%Box-Jenkins Series A
%'armax',na=1,nb=0,nc=1,delay=0.
%'arma',nc=1,nd=0,diff=1.
%'arma',nc=1,nd=1,diff=0.
%'arma',nc=0,1,nd=0,1,diff=0,1.

%Box-Jenkins Series C
%'arma',nc=0,nd=1,diff=1.
%'arma',nc=2,nd=0,diff=2.
%'arma',nc=0-2,nd=0-1,diff=0-2.

%Box-Jenkins Series J
'bjtf',nb=2,nc=0,nd=2,nf=1,delay=3,diff=0.
%'bjtf',nb=1-2,nc=0-1,nd=1-2,nf=1-2,delay=2-3,diff=0.
%'armax',na=3,nb=4,nc=1,diff=0,per=[],delay=3.
	%arma for uj
%'arma',nc=0,nd=3,diff=0. 
   %arma for yj
%'arma',nc=2,nd=4,diff=0. 

%Box-Jenkins Series M
%'arma',nc=1,nd=0,diff=1.
%'bjtf',nb=0,nc=1,nd=0,nf=1,delay=3,diff=1.

