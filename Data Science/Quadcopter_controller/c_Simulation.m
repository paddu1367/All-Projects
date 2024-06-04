clear; clc;

% For simulation purposes
M = 1; L = 1; g = 9.81; mu = 0.01;
delta = 0.1; 
load gains;

tspan = 0:delta:5;
x = [0.1, 0.1]'; % Initial condition

for k = 1:length(tspan)-1
    u(:, k) = K*x(:, k);
    x(1, k+1) = x(1, k) + delta*x(2, k);
    x(2, k+1) = delta*g/L*sin(x(1, k)) + ...
                   (1 - delta*mu/M/L^2)*x(2, k) + ...
                   delta/M/L^2*u(:, k);
end

subplot(2,1,1)
plot(tspan, x(1, :));

subplot(2,1,2)
plot(tspan, x(2, :));