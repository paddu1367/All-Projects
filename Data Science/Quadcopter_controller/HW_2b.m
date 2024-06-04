clear; clc;
%loading the data
load K
load model
%assigning values
tsim= 0:0.01:5;
k1=K(1);
k2=K(2);
r=1;

xsim = [0; 0];

%stimulation of system
for k = 2:length(tsim)
    u = k1*(xsim(1,k-1) -r) + k2*xsim(2,k-1);
    xsim(:,k) =model*[u; xsim(:, k-1)];

end


x1sim = xsim(1, :);
x2sim = xsim(2, :);

cla; clf;
%plot of simulated results and the validation results
subplot(2,1,1)
plot(tsim, x1sim);
title('plot of x1');

subplot(2,1,2)
plot(tsim, x2sim);
title('plot of x2');