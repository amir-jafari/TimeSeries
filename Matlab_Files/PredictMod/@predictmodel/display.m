function display(pmod)
%DISPLAY Display the name and properties of a prediction model variable.
%
%	Syntax
%
%	  display(pmod)
%
%	Description
%
%	  DISPLAY(PMOD) displays a prediction model variable's name and properties.
%
%	Examples
%
%	  Here a perceptron variable is defined and displayed.
%
%	    pmod = newbjtf(1,1,1,1);
%	    display(pmod)
%
%	  DISPLAY is automatically called as follows:
%
%	    pmod
%
%	See also DISP

%	07-24-00
% $Revision: 1.0 $

isLoose = strcmp(get(0,'FormatSpacing'),'loose');

if (isLoose), fprintf('\n'), end

fprintf('%s =\n',inputname(1));

disp(pmod)
