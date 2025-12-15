function [yacf,ypacf,ygpac] = uniAnal(y,na,np,nrg,ncg,diff,per,perdsp)
%UNIANAL Univariate analysis
%	
%	Syntax
%
%	  [yacf,ypacf,ygpac] = uniAnal(y,na,np,nrg,ncg,diff,per,perdsp)
%
%	Description
%	
%	  UNIANAL computes the autocorrelation and partial
%	  autocorrelation functions and the generalized partial
%	  autocorrelation array of the sequence y and
%	  plots the results.
%	
%	  UNIANAL(Y,NA,NP,NRG,NCG,DIFF,PER,PERDSP) takes these inputs,
%	    Y   - Vector containing the sequence.
%	    NA  - The autocorrelation lags will range from -NA to +NA; default = 20.
%	    NP  - Number of partial autocorrelation terms to compute; default = 10.
%	    NRG - Number of rows of the GPAC to compute; default = 5.
%	    NCG - Number of columns of the GPAC to compute; default = 5.
%	    DIFF  = [diff1 diff2...diffNP], default = [0].
%	      diffi - Order of the differencing for period i.
%	    PER   = [per1 per2...perNP], default = [].
%	      peri  - Period i.
%	    PERDSP- Period at which to display acf and pacf data.; default = 1;
%	  and returns,
%	    YACF  - Autocorrelation function for Y.
%	    YPACF - Partial autocorrelation function for Y.
%	    YGPAC - Generalized partial autocorrelation function for Y.
%		
%	Examples
%
%	  This code generates an autoregressive sequence.
%	
%	    e = randn(1,2000);
%	    y = filter(1,[1 -.8],e);
%
%	  The following command generates the autocorrelation and
%	  partial autocorrelation functions.  The acf will be
%	  computed from lag -20 to lag 20.  The pacf will be computed
%	  from order 1 to order 10. 
%
%	    [yacf,ypacf,ygpac] = uniAnal(y,20,10);
%

% Yong Hu, Martin Hagan, 9-15-00
% $Revision: 1.0 $ $Date: 21-Sep-2000 14:37:36 $

if nargin<2, na = 20;end
if nargin<3, np = 10;end
if nargin<4, nrg = 5;end
if nargin<5, ncg = nrg;end
if nargin<6, diff = 0;end
if nargin<7, per = [];end
if nargin<8, perdsp = 1;end

numpts = length(y);

if (length(per)+1) ~= length(diff),
  error('per must have one less term than diff.')
end

% Difference the sequence
period = [1 per];
for i=1:length(diff),
  d = diff(i);
  if (d~=0),
    y = sdiff(y,d,period(i));
  end
end

tot = max([np+1 nrg+ncg]);
if na<tot,
  L = tot*perdsp;
else
  L = na*perdsp;
end

if L>numpts,
  error('Not enough data to compute the acf for sufficient lags.');
end

yacf = xcorr(y,y,L,'unbiased');
len = length(yacf);

% Select data only at multiples of the display period
if perdsp>1,
  yacf = [fliplr(yacf((L+1-perdsp):-perdsp:1)) yacf((L+1):perdsp:len)];
end

% Calculate pacf and gpac
ypacf = parcor(yacf,np);
[ygpac]=gpac(yacf,nrg,ncg);

% Convert acf to proper length for plotting
len = length(yacf);
L = round((len-1)/2);
if L~=na,
  dif = L-na;
  yacf = yacf(1+dif:len-dif);
end

lag = -na:na;

confint = yacf(na+1)*2/sqrt(numpts);

% Get figure if it exists, or create new figure
me = 'ACF and PACF';
fig1 = chkfig(me);
if length(get(fig1,'children')) == 0, fig = 0; end
if (exist('fig1')&&(fig1~=0))
  figure(fig1);
else
  fig1 = figure;
  set(fig1,'name',me,'numbertitle','off')
end

subplot(2,1,1), bar(lag,yacf,.4)
title('Autocorrelation Function')
xlabel('Lag')
hold on
ax = axis;
plot([ax(1) ax(2)],[0 0],'k');
plot([ax(1) ax(2)],[confint confint],':r');
plot([ax(1) ax(2)],[-confint -confint],':r');
hold off
subplot(2,1,2), bar(ypacf,.4)
title('Partial Autocorrelation Function')
xlabel('Lag')
hold on
ax = axis;
plot([ax(1) ax(2)],[0 0],'k');
hold off

% Get figure if it exists, or create new figure
me = 'GPAC Array';
fig2 = chkfig(me);
if length(get(fig2,'children')) == 0, fig = 0; end
if (exist('fig2')&&(fig2~=0))
  figure(fig2);
else
  fig2 = figure;
  set(fig2,'name',me,'numbertitle','off')
end

plotgpac(ygpac,'GPAC Array');
