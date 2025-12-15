function [g,rv,g_gpac,h_gpac] = multiAnal(u,y,nng,ndg,nnh,ndh,lg,lh)
%MULTIANAL Multi-variable analysis.
%	
%	Syntax
%
%	  [g,rv,g_gpac,h_gpac] = multiAnal(u,y,nng,ndg,nnh,ndh,lg,lh)
%
%	Description
%	
%	  MULTIANAL provides an analysis between two sequences: u and y.
%	  It estimates the impulse response beteen the two variables, the
%	  residual autocorrelation function, the GPAC for the G transfer
%	  function between u and y, and the GPAC for the H transfer function
%	  between e and y (ARMA model). 
%	
%	  MULTIANAL(U,Y,NNG,NDG,NNH,NDH,LG,LH) takes these inputs,
%	    U   - Input sequence.
%	    Y   - Output sequence.
%	    NNG - Maximum order for G transfer function numerator; default = 5.
%	    NDG - Maximum order for G transfer function denominator; default = 5.
%	    NNH - Maximum order for H transfer function numerator; default = 5.
%	    NDH - Maximum order for H transfer function denominator; default = 5.
%	    LG  - Number of lags of the impulse response to compute; default = 20.
%	    LH  - Number of lags of the residual acf to compute; default = 20.
%	  and returns,
%	    G      - Estimated impulse response between u and y.
%	    RV     - Residual autocorrelation function.
%	    G_GPAC - GPAC for the G transfer function.
%	    H_GPAC - GPAC for the H transfer function.
%		
%	Examples
%
%	  This code generates a first order sequence y from an
%	  input sequence u and a noise sequence e.
%	
%	    e = randn(1,2000)*0.2;
%	    u = randn(1,2000);
%	    y = filter(1,[1 .5],u) + filter(1,[1 -.8],e);
%
%	  The following command generates the multivariate analysis 
%	  for the u and y sequences.  It uses the default orders of 5.
%      
%	    [g,rv,g_gpac,h_gpac] = multiAnal(u,y);
%

% Yong Hu, Martin Hagan, 9-15-00
% $Revision: 1.0 $ $Date: 21-Sep-2000 14:37:36 $


if nargin<3, nng = 5;end
if nargin<4, ndg = nng;end
if nargin<5, nnh = 5;end
if nargin<6, ndh = nnh;end
if nargin<7, lg = 20;end
if nargin<8, lh = 20;end

%K = nng + ndg + 1;
K = lg;
[g] = impest(u,y,K);
index = 0:K;

% Get figure if it exists, or create new figure
me = 'Impulse Response and Residual ACF';
fig1 = chkfig(me);
if length(get(fig1,'children')) == 0, fig = 0; end
if (exist('fig1')&&(fig1~=0))
  figure(fig1);
else
  fig1 = figure;
  set(gcf,'name',me,'numbertitle','off')
end

subplot(2,1,1), bar(index,g,0.4);
title('Impulse Response')
xlabel('Lag')
hold on
ax = axis;
plot([ax(1) ax(2)],[0 0],'k');
hold off

% Get figure if it exists, or create new figure
me = 'G & H GPAC Arrays';
fig2 = chkfig(me);
if length(get(fig2,'children')) == 0, fig = 0; end
if (exist('fig2')&&(fig2~=0))
  figure(fig2);
else
  fig2 = figure;
  set(gcf,'name',me,'numbertitle','off')
end

g_aug = [zeros(K,1);g];
[g_gpac]=gpac(g_aug,nng,ndg);
subplot(2,1,1), plotgpac(g_gpac,'GPAC for G Transfer Function');

y1 = conv(g,u);
yy1 = filter(g,1,u);
y1 = y1(1:length(y));

v = y - y1;

L = nnh + ndh + 1;
rv = xcorr(v,v,L,'unbiased');
lag = -L:L;
confint = rv(L+1)*2/sqrt(length(v));

figure(fig1)
subplot(2,1,2), bar(lag,rv,0.4);
title('Autocorrelation Function for V')
xlabel('Lag')
hold on
ax = axis;
plot([ax(1) ax(2)],[0 0],'k');
plot([ax(1) ax(2)],[confint confint],':r');
plot([ax(1) ax(2)],[-confint -confint],':r');
hold off

figure(fig2)
[h_gpac] = gpac(rv,nnh,ndh);
subplot(2,1,2), plotgpac(h_gpac,'GPAC for H Transfer Function');

