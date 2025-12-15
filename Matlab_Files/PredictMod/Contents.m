% Prediction Model Toolbox.
% Version 1.0be  15-Sept-2000
%
% Analysis functions.
%   calcindex - Calculate the performance index.
%   gpac      - Compute the GPAC for a given autocorrelation function.
%   impest    - Estimate impulse response between two time series.
%   jacobian  - Compute the jacobian matrix of prediction errors.
%   multianal - Impulse response, residual ACF and GPAC's for dual time series.
%   multichi  - Chi-square test for dual time series.
%   parcor    - Compute the partial autocorrelation function.
%   pmodaic   - AIC for a fitted model.
%   pmodbic   - BIC for a fitted model.
%   unianal   - ACF, PACF and GPAC for single time series.
%   unichi    - univariate Chi-square test.
%
% Estimation functions.
%   estimate - General parameter estimation framework.
%   estimlm  - Levenberg-Marquardt optimization.
%
% Getting and setting model parameters.
%   getmx      - Get the parameters for a prediction model.
%   getmxarma  - Get the parameters for an ARMA model.
%   getmxarmax - Get the parameters for an ARMAX model.
%   getmxarx   - Get the parameters for an ARX model.
%   getmxbjtf  - Get the parameters for a Box and Jenkins transfer function model.
%   getmxregr  - Get the parameters for a linear regression model.
%   setmx      - Set the parameters of a prediction model.
%   setmxarma  - Set the parameters of an ARMA model.
%   setmxarmax - Set the parameters of an ARMAX model.
%   setmxarx   - Set the parameters of an ARX model.
%   setmxbjtf  - Set the parameters of a Box and Jenkins transfer function model.
%   setmxregr  - Set the parameters of a linear regression model.
%
% Initializing model parameters.
%   initrand  - Initialize model parameters to uniformly distributed random values.
%   initrandn - Initialize model parameters to normally distributed random values.
%   initzero  - Initialize model parameters to zeros.
%
% New prediction models.
%   newarma  - Create an ARMA model.
%   newarmax - Create an ARMAX model.
%   newarx   - Create an ARX model.
%   newbjtf  - Create a Box and Jenkins transfer function model.
%   newregr  - Create a linear regression model.
%
% Performance index functions.
%   pmodmse  - Mean squared error performance function.
%
% Plotting functions.
%   plotgpac  - Graphical display of GPAC array.
%   plotindex - Plot performance index during parameter estimation.
%
% Pre and Post Processing.
%   sdiff     - Difference a time series.
%
% Prediction functions.
%   predarma - Prediction for ARMA models.
%   predarmax - Prediction for ARMAX models.
%   predarx   - Prediction for ARX models.
%   predbjtf  - Prediction for Box and Jenkins transfer function models.
%   predregr  - Prediction for linear regression models.
%
% Reformatting models.
%   getGH      - Put prediction model in form of G and H transfer functions.
%   getGHarma  - Put ARMA model in form of G and H transfer functions.
%   getGHarmax - Put ARMAX model in form of G and H transfer functions.
%   getGHarx   - Put ARX model in form of G and H transfer functions.
%   getGHbjtf  - Put BJTF model in form of G and H transfer functions.
%
% Selecting models
%   selpmod - select best prediction model from a class of possible models.
%
% Testing functions (most in pmodtest subdirectory).
%   pmsample    - Sample modeling session.
%   spec        - Example specification file for model selection function selpmod.
%   testaic     - Test the pmodaic and pmodbic functions.
%   testarma    - Test ARMA  modeling.
%   testarmax   - Test ARMAX modeling.
%   testarx     - Test ARX modeling.
%   testbjtf    - Test Box and Jenkins transfer function modeling.
%   testprepost - Test pre- and post- processing.
%   testregr    - Test linear regression modeling.
%   testselpmod - Test model selection.
%
% Using prediction models.
%   estimate  - Estimate the parameters of a prediction model.
%   pmodsim   - Simulate a prediction model.
%   predict   - Use a prediction model to predict a time series.
%
% Utility functions.
%	 chisqrdf - Calculate Chi square cumulative density function
%   chkfig   - Check to see if a figure already exists.
%   cliprec  - Clip estimation record to those iterations that were performed.
%   gcombvec - Generalized vector combinations.
%   makerow  - Convert matrix so that it contains more columns than rows.
%   newrec   - Initialize an estimation record structure.
%   sepym    - Extract output and objective function weight matrix from a structure.
%   setfig   - Set up a figure.
%

% $Revision: 1.0 $ $Date: 21-Sep-2000 14:37:36 $
