%PMSAMPLE Sample Prediction Model Session

% Yong Hu, Martin Hagan, 9-15-00
% $Revision: 1.0 $ $Date: 21-Sep-2000 14:37:36 $

echo on
clc
%    ==========================================================
%    PMSAMPLE        
%    ==========================================================

%    estimate  - General parameter estimation framework.
%    multianal - Impulse response, residual ACF and GPAC's for dual time series.
%    newbjtf   - Create a Box and Jenkins transfer function model.
%    newregr   - Create a linear regression model.
%    pmodaic   - AIC for a fitted model.
%    predict   - Use a prediction model to predict a time series.
%    selpmod   - Select the best prediction model.
%    unianal   - ACF, PACF and GPAC for single time series.

%    TIME SERIES ANALYSIS:

%    Prediction models are developed by the above functions
%    for two different set of time series.  One set consists of
%    a gas furnace data (input: u, gas rate in cubic feet per minute;
%    output: y, % CO2 in outlet gas).  The second set consists of
%    simulated data.

pause % Strike any key to continue...
clc

%    DEFINING THE  PROBLEM
%    =====================

%    The .mat file FURNACE contains vectors of u and y. 
%    The u vector contains the prediction model inputs, 
%    gas rate in cubic feet per minute.  The y vector contains
%    the corresponding desired model ouputs, %CO2 in outlet gas.

% Load in the data file and make both input and output zero mean;

load furnace;
u = u-mean(u);
y = y-mean(y);

% Plot data

me = 'PMSAMPLE';
fig1 = setfig(me);
subplot(2,1,1), plot(u)
title('Input Sequence')
subplot(2,1,2), plot(y)
title('Output Sequence')

pause % Strike any key to analyze the data...
clc

% Find the acf, pacf and gpac for the u sequence.

[uacf,upacf,ugpac] = uniAnal(u,20,10);

% It appears that sequence u might be estimated by an AR
% process.  Select the figure labeled "ACF and PACF" to see
% the autocorrelation function and partial autocorrelation
% function.  The autocorrelation function is infinite, like
% damped exponentials.  The partial autocorrelation function
% approximately cuts off after lag 3.  The figure labeled
% "GPAC Array" displays the GPAC for this data. The first row
% contains small values begining in the fourth column,
% also indicating it might be an AR process.

pause % Strike any key to continue analysis...
clc

% Find the impulse response between u and y, the residual
% autocorrelation, and the GPAC array for each.  In the 
% figure labeled 'Impulse Response and Residual ACF', the
% impulse response shows 3 delays between u and y.  From the
% figure named 'G & H GPAC Array', the H gpac showed that H may 
% be approximated with numerator order of 0 and denominator
% order of 2.

[g,rv,g_gpac,h_gpac] = multiAnal(u,y,6,6);

pause % Strike any key to model the data...
clc

% For the first prediction model of this data we will
% use the Box and Jenkins transfer function model.  The
% model is in the form
%    y(t) = G u(t) + H e(t)
% where G is a transfer function with numerator pmod.b
% and denominator pmod.f, H is a transfer function with
% numerator pmod.c and denominator pmod.d.  String SPEC
% specifies a BJTF to be estimated, and possible
% orders of numerators and denominators in G and H.  
 
spec = '''bjtf'',nb=1-2,nc=0,nd=2,nf = 0-1,delay=3,diff=0.';
estpmod = selpmod(spec,y,u);

pause % Strike any key to select the best model...
clc

% select the best BJTF model based on BIC criteria.
pmod = estpmod.bjtf.bicmod;

pause % Strike any key to display the results...
clc

%    TESTING THE PREDICTION MODEL
%    ===========================

y2 = predict(pmod,y,u);
ind = 1:length(y2);
fig1 = setfig(me);
plot(ind,y,'o',ind,y2,'x')
legend('y(t)','yhat(t)')

% These are the final parameter values

pmod.b{1}
pmod.f{1}
pmod.c{1}
pmod.d{1}

pause % Strike any key to start second test...
clc
pause(1);
clc

% For the second prediction model of this data we will
% use the ARMAX model.  The model is in the form
%    y(t) = G u(t) + H e(t)
% where G is a transfer function with numerator pmod.b
% and denomitor pmod.a, H is a transfer function with
% numerator pmod.c and denominator pmod.a.  String SPEC
% specifies an ARMAX to be estimated, and possible
% orders of numerators and denominators in G and H.  
 
spec = '''armax'',na=1-3,nb=3-4,nc=0-1,delay=3,diff=0.';
estpmod = selpmod(spec,y,u);

pause % Strike any key to select the best model...
clc

% select the best ARMAX model based on AIC criteria.
pmod = estpmod.armax.aicmod;

% By careful examination of the ARMAX and BJTF model
% structure and parameters, the two estimated models
% are equivalent.

pause % Strike any key to display the results...
clc
%    TESTING THE PREDICTION MODEL
%    ===========================

y2 = predict(pmod,y,u);
ind = 1:length(y2);
fig1 = setfig(me);
plot(ind,y,'o',ind,y2,'x')
legend('y(t)','yhat(t)')

% These are the final parameter values

pmod.a{1}
pmod.b{1}
pmod.c{1}

pause % Strike any key to start third test...
clc

% For the third prediction model of this data we will
% use the linear regression model.  The function NEWREGR
% creates the model structure.  Here we have a single
% variable regression, since there is only one row in u.
% pmod.b contains the regression parameters.

pmod = newregr(1);

pause % Strike any key to estimate the model parameters...
clc

% We estimate the model parameters for 50 iterations
% but the regression model will always converge after
% one iteration.

pmod.estimParam.epochs = 50;
pmod.estimParam.show = 20;
pmod = estimate(pmod,y,u);

pause % Strike any key to display the results...
clc

%    TESTING THE PREDICTION MODEL
%    ===========================

y2 = predict(pmod,y,u);
ind = 1:length(y2);
fig1 = setfig(me);
plot(ind,y,'o',ind,y2,'x')
legend('y(t)','yhat(t)')

% These are the final parameter values.  Since u and y
% are zero mean, the first parameter is very close to
% zero.

pmod.b{1}

pause % Strike any key to start fourth test...
clc
pause(1);
clc

%%
% For the fourth prediction model of this data we will
% use the ARMA model.  The model is in the form
%    u(t) = H e(t)
% where H is a transfer function with numerator pmod.c
% and denomitor pmod.d.  String SPEC specify an ARMA
% to be estimated, and possible orders of numerator
% and denominator in H.  
 
u=ym;
spec = '''arma'',nc=0-2,nd=1-4,diff=0.';
estpmod = selpmod(spec,u);

pause % Strike any key to select the best model...
clc

% select the best ARMA model based on BIC criteria.
pmod = estpmod.arma.bicmod;

pause % Strike any key to display the results...
clc
%    TESTING THE PREDICTION MODEL
%    ===========================

u2 = predict(pmod,u);
ind = 1:length(u2);
fig1 = setfig(me);
plot(ind,u,'o',ind,u2,'x')
legend('u(t)','uhat(t)')

% We can also perform a chi-square test on the residuals
% of the fitted model with the function unichi.  If the
% returned variable pass = 1, then the model is adequate
% at the 95% confidence level.

pass = unichi(pmod,u);

% These are the final parameter values

pmod.c{1}
pmod.d{1}


%%
pause % Strike any key to continue the second problem...
clc
%    DEFINING THE  SECOND PROBLEM
%    =============================

%    Here we create some simulated data from a known model.  
%    The u matrix contains the prediction model inputs,
%    and the y matrix contains the corresponding desired model  
%    ouputs.

% Create the data

e = randn(1,2000)*0.2;
u = randn(1,2000);
y = filter([0 2],[1 .5],u) + filter([1 .25],[1 -.8],e);

% Plot the data

fig1 = setfig(me);
subplot(2,1,1), plot(u)
title('Input Sequence')
subplot(2,1,2), plot(y)
title('Output Sequence')

pause % Strike any key to analyze the data...
clc

% Find the impulse response between u and y, the residual
% autocorrelation, and the GPAC array for each.  In the
% figure labeled "Impulse Response and Residual ACF" the
% impulse response appears to a simple damped oscillation.
% This indicates a possible first order model with a 
% negative parameter.  The first nonzero value of the
% impulse response is at a lag of one, which indicates
% a delay of one in the model.  The GPAC for the G
% transfer function verifies this analysis.  The large
% values across the first row indicate the delay of one.
% The sequence of constant values in the first column
% after the first row indicate a first order 
% denominator.

[g,rv,g_gpac,h_gpac] = multiAnal(u,y);

pause % Strike any key to model the data...
clc

% For the prediction model of this data we will.
% use the Box and Jenkins transfer function model.  The
% function NEWBJTF creates the model structure.  Here
% we have first order numerators and denomitors for
% each transfer function, except for the zero order
% numerator for the G transfer function.

pmod = newbjtf(0,1,1,1,1);


pause % Strike any key to estimate the model parameters...
clc

% We estimate the model parameters for 50 iterations
% and displaying the results every 20 iterations.

pmod.estimParam.epochs = 50;
pmod.estimParam.show = 20;
[pmod,trec,stat] = estimate(pmod,y,u);

pause % Strike any key to display the results...
clc
%    TESTING THE PREDICTION MODEL
%    ===========================

y2 = predict(pmod,y,u);
ind = 1:length(y2);
fig1 = setfig(me);
plot(ind,y,'o',ind,y2,'x')
legend('y(t)','yhat(t)')

% These are the final parameter estimates, their standard deviations
% and the true values.

[pmod.b{1} pmod.f{1} pmod.c{1} pmod.d{1}]
[stat.stdx' ]
[ 2  0.5 .25 -.8]

% This is the AIC of the fitted model.  This can be used
% to compare different models.

aic = pmodaic(pmod,y,u)

% We can also perform chi-square tests on the fitted
% model with the function multichi.  If the pass
% array is [1 1], then the fitted model is adequate
% at the 95% confidence level.

[pass] = multichi(pmod,y,u)



pause % Strike any key to continue the sample session...
clc

% We can also weight the data so that errors that 
% occur near the end of the data are emphasized more
% than errors that occur at the beginning of the
% data.  For this we use a weighting vector m.
% In this example we create data from two different
% models.  We then weight the data higher at the end
% of the data set.  The resulting errors at the end
% of the data set are smaller.

e1 = randn(1,1000)*0.2;
u1 = randn(1,1000);
y1 = filter([0 2],[1 .5],u1) + filter([1 .25],[1 -.8],e1);

e2 = randn(1,1000)*0.2;
u2 = randn(1,1000);
y2 = filter([0 1],[1 .9],u1) + filter([1 .7],[1 -.2],e1);

y = [y1 y2];
u = [u1 u2];

pmod = newbjtf(0,1,1,1,1);

m = 0:length(y)-1;
m = 0.99.^m;
m = fliplr(m);
ystr.y = y;
ystr.m = m;
pmod.estimParam.show = 20;
[pmod,trec,stat] = estimate(pmod,ystr,u);

pause % Strike any key to test the new model...
clc

% We can see that the errors at the end of the
% data set are smaller.

y2 = predict(pmod,y,u);
ind = 1:length(y2);
fig1 = setfig(me);
plot(ind,y-y2,'x')
legend('Prediction Error')

echo off
disp('End of PMSAMPLE')

