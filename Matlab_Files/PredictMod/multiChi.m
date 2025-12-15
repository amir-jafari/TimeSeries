function [pass,q,s,nq,ns] = multiChi(pmod,y,u,k1,k2,alpha1,alpha2)
%MULTICHI Multivariate Chi-Square test
%	
%	Syntax
%
%	  [pass,q,s,nq,ns] = multiChi(pmod,y,u,k1,k2,alpha1,alpha2)
%
%	Description
%	
%	  MULTICHI performs multivariate chi-square tests. It calculates
%    two chi-square statistics.  The first statistic tests the
%    whiteness of the residuals.  The second tests crosscorrelations
%    between residuals and model inputs.
%	
%	  MULTICHI(PMOD,Y,U,K1,K2,ALPHA1,ALPHA2) takes these inputs,
%	    PMOD   - Prediction model.
%	    Y      - Prediction model desired outputs.
%	    U      - Prediction model inputs.
%	    K1     - Number of lags of the ACF to use, default = 20.
%	    K2     - Number of lags of the cross correlation to use,default = 20.
%	    ALPHA1 - Probability of type I error for the residuals, default = 0.05.
%	    ALPHA2 - Probability of type I error for the crosscorrelation, default = 0.05.
%	  and returns,
%	    PASS - Pass(1) = 1, if test for the residual correlation is passed, 0 otherwise;
%	           Pass(2) = 1, if test for the cross correlation is passed, 0 otherwise.
%	    Q    - The chi-square statistic for the residual autocorrelation.
%	    S    - The chi-square statistic for the crosscorrelation.
%	    NQ   - Degrees of freedom for Q.
%	    NS   - Degrees of freedom for S.
%		
%	Examples
%
%	  This code estimates a Box-Jenkins Transfer Function model from the
%	  gas furnace data.
%	
%	    load furnace;
%	    y = y-mean(y);
%	    u = u-mean(u);
%	    pmod = newbjtf(2,0,2,2,3,0);
%	    pmod.estimParam.show = NaN;
%	    [pmod,trec,stat] = estimate(pmod,y,u);
%      
%	  The estimated BJTF model then performs chi-square test.
%
%	    [pass,q,s,nq,ns] = multichi(pmod,y,u)
%
%	See also UNICHI.

% Yong Hu, Martin Hagan, 9-15-00
% $Revision: 1.0 $ $Date: 21-Sep-2000 14:37:36 $

if nargin < 3
  error('Not enough input arguments')
end
if nargin<4, k1 = 20; end
if nargin<5, k2 = 20; end
if nargin<6, alpha1 = 0.05; end
if nargin<7, alpha2 = 0.05; end

%find prewhitened input al
disp('prewhitening input, please wait...');
estpmodu = selpmod('''arma'',nc=0-3,nd=0-3,diff=0.',u);
Rq = [1 estpmodu.arma.bicmod.c{1}];
Sq = [1 estpmodu.arma.bicmod.d{1}];
al = filter(Sq,Rq,u);

be = filter(Sq,Rq,y);
eps = be - filter([zeros(1,pmod.delay) pmod.b{1}],[1 pmod.f{1}],al);


%AUTO CORRELATION TEST
e = y - predictdf(pmod,y,u);
%e = eps;
%auto correlation function of residual error
acf = xcorr(e,e,k1,'unbiased');
acf(1:(length(acf)-1)/2) = [];
acf = acf/acf(1);
acf(1) = [];

q = length(y)*sum(acf.^2);
if isempty(pmod.c)
   nc = 0;
else
   nc = length(pmod.c{1});
end
if isempty(pmod.d)
   nd = 0;
else
   nd = length(pmod.d{1});
end
nq = k1-(nc+nd);
prq = chisqrdf(q,nq);
if (1-prq)>alpha1
   pass(1) = 1;
else
   pass(1) = 0;
end



%CROSS CORRELATION TEST


%auto correlation function of cross correlation
%ccf = xcorr(al,eps,k2,'unbiased');
ccf = xcorr(al,e,k2,'unbiased');
ccf(1:(length(ccf)-1)/2) = [];
alvar = var(al);
%epsvar = var(eps);
%ccf = ccf/sqrt(alvar*epsvar);
epsvar = var(e);
ccf = ccf/sqrt(alvar*epsvar);

s = length(y)*sum(ccf.^2);
if isempty(pmod.b)
   nb = 0;
else
   nb = length(pmod.b{1})-1;
end
if isempty(pmod.f)
   nf = 0;
else
   nf = length(pmod.f{1});
end
ns = k2+1-(nb+nf+1);
prs = chisqrdf(s,ns);
if (1-prs)>alpha2
   pass(2) = 1;
else
   pass(2) = 0;
end




















