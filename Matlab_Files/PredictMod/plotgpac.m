function plotgpac(gpac,gtitle)
%PLOTGPAC Plot the gpac array.
%
%	Syntax
%
%	  PLOTGPAC(GPAC,GTITLE)
%
%	Description
%	
%	  PLOTGPAC(GPAC,GTITLE) takes these inputs,
%	    GPAC   - GPAC array.
%	    GTITLE - Title for the plot.
%	  and displays the GPAC array represented as a grid of squares.
%	
%	  Each square's AREA represents the magnitude of an element.
%	  Each square's COLOR represents the element's sign.
%	  RED for negative values, GREEN for positive.
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
%	    acf = xcorr(y,y,20,'unbiased');
% 	    [gpac_array] = gpac(acf,7,7);
%	    plotgpac(gpac_array);
%	    
%

% Mark Beale, 1-31-92
% Revised 12-15-93, MB
% Revised 11-31-97, MB
% Revised from network weights to GPAC display 7-1-00
% Martin Hagan, 9-15-00
% $Revision: 1.0 $ $Date: 21-Sep-2000 14:37:36 $
% $Revision: 1.1 $ $Date: 12-Dec-2006 16:44:39 $ use ind instead of [i,j]
% to find min and max

if nargin < 1,error('Not enough input arguments.');end
if nargin < 2, gtitle = 'GPAC Array';end
limit = 5;
[ind] = find(gpac>limit);
if (~isempty(ind)),
  gpac(ind) = limit;
end
[ind] = find(gpac<-limit);
if (~isempty(ind)),
  gpac(ind) = -limit;
end

%max_m = max(max(abs(gpac)));
max_m = limit;
min_m = max_m / 300;

% DEFINE BOX EDGES
xn1 = [-1 -1 +1]*0.5;
xn2 = [+1 +1 -1]*0.5;
yn1 = [+1 -1 -1]*0.5;
yn2 = [-1 +1 +1]*0.5;

% DEFINE POSITIVE BOX
xn = [-1 -1 +1 +1 -1]*0.5;
yn = [-1 +1 +1 -1 -1]*0.5;

% DEFINE POSITIVE BOX
xp = [xn [-1 +1 +1 +0 +0]*0.5];
yp = [yn [+0 +0 +1 +1 -1]*0.5];

[S,R] = size(gpac);

cla reset
hold on
set(gca,'xlim',[0 R]+0.5);
set(gca,'ylim',[0 S]-0.5);
set(gca,'xlimmode','manual');
set(gca,'ylimmode','manual');
xticks = get(gca,'xtick');
set(gca,'xtick',xticks(find(xticks == floor(xticks))))
yticks = get(gca,'ytick');
set(gca,'ytick',yticks(find(yticks == floor(yticks))))
set(gca,'ydir','reverse');
if get(0,'screendepth') > 1
  set(gca,'color',[1 1 1]*.5);
  %set(gcf,'color',[1 1 1]*.3);
end

for i=1:S
  i1 = i-1;
  for j=1:R
    m = sqrt((abs(gpac(i,j))-min_m)/max_m);
	m = min(m,max_m)*0.95;
	if real(m)
	  if gpac(i,j) >= 0
	    fill(xn*m+j,yn*m+i1,[0 0.8 0])
        plot(xn1*m+j,yn1*m+i1,'w',xn2*m+j,yn2*m+i1,'k')
	  elseif gpac(i,j) < 0
	    fill(xn*m+j,yn*m+i1,[0.8 0 0]);
        plot(xn1*m+j,yn1*m+i1,'k',xn2*m+j,yn2*m+i1,'w');
	  end
	end
  end
end

plot([0 R R 0 0]+0.5,[0 0 S S 0]-0.5,'w');
xlabel('Denominator Order');
ylabel('Numerator Order');
title(gtitle);
grid on
