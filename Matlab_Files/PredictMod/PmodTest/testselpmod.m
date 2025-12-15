% This file is a collection of script which is intended 
% to test the selpmod function.
%
% file 'spec.m' is used to provide specifications.  In order
% to run the script, some lines in 'spec.m' need to be 
% 'uncommented'.


%ARX
%Copy the following code to MATLAB command window.
%Generate y,u by an ARX model.
nb = 1; 
na = 2;
delay = 1;
b = {[2 3]};
a = {[-1 0.25]};
u=randn(1,2000);
e=randn(1,2000)*.5;
pmodr = newarx(nb,na,delay);
pmodr.a = a;
pmodr.b = b;
y = pmodsim(pmodr,e,u);   
%run selpmod
estpmod = selpmod('spec.m',y,u);
    
%ARMA
nc = 2; 
nd = 2;
c = {[0.5 0.8]};
d = {[-1 0.25]};
e = randn(1,2000)*.5;
pmodr = newarma(nc,nd);
pmodr.c = c;
pmodr.d = d;
y = pmodsim(pmodr,e);
estpmod = selpmod('spec.m',y);

%ARMAX
na = 2;
nb = 1; 
nc = 1;
delay = 1;
a = {[-1 0.25]};
b = {[2 3]};
c = {[0.65]};
u=randn(1,2000);
e=randn(1,2000)*.5;
pmodr = newarmax(na,nb,nc,delay);
pmodr.a = a;
pmodr.b = b;
pmodr.c = c;
y = pmodsim(pmodr,e,u);
estpmod = selpmod('spec.m',y,u);

%BJTF
nb = [1 1];
nc = 1;
nd = 2;
%nf = [1 2];
nf = [1 1];
%delay = [1 2];
delay = [1 1];
diff = [0];
per = [];
%f = {[0.35] [-1  .3]};
b = {[1 1] [2 3]};
c = {[0.65]};
d = {[-1 0.25]};
f = {[0.35] [-1]};
u=randn(2,2000);
e=randn(1,2000)*.5;
pmoda = newbjtf(nb,nc,nd,nf,delay,diff,per);
pmoda.f = f;
pmoda.d = d;
pmoda.c = c;
pmoda.b = b;
y = pmodsim(pmoda,e,u);
estpmod = selpmod('spec.m',y,u);

%REGR
nb = 3; 
delay = [1 0 1];
b = {[1.2 0.5 2.0 0.8]};
u = randn(nb,2000);
e = randn(1,2000)*.5;
pmodr = newregr(nb,delay);
pmodr.b = b;
pmodr.delay = delay;
y = pmodsim(pmodr,e,u);
estpmod = selpmod('spec.m',y,u);

















     
