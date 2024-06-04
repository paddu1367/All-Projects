clear; clc;
%loading the data
load dataval
load model
%assigning values
tval = logval(:, 1)';
uval = logval(:, 2)';
x1val = logval(:, 3)';
x2val = logval(:, 4)';

tsim= tval;
x1sim(1) = x1val(1);
x2sim(1) = x2val(1);
xsim = [x1sim(1); x2sim(1)];

%stimulation of system
for k = 2:length(tsim)
    xsim(:,k) =model*[uval(k-1); xsim(:, k-1)];

end


x1sim = xsim(1, :);
x2sim = xsim(2, :);

cla; clf;
%plot of simulated results and the validation results
subplot(2,1,1)
plot(tsim, x1val, '--black'); hold on;
plot(tsim, x1sim);
title('plot of x1');
hold off

subplot(2,1,2)
plot(tsim, x2val, '--black'); hold on;
plot(tsim, x2sim);
title('plot of x2');