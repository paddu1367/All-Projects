clear; clc;

syms x1 x2 u real;
syms delta g m L mu real;

x = [x1; x2];

n = length(x);
m = length(u);

f1 = x1 + delta*x2;
f2 = delta*g/L*sin(x1) + (1 - delta*mu/m/L^2)*x2 + delta/m/L^2*u;


f = [f1; f2];

JA = jacobian(f, x);
JB = jacobian(f, u);

A = subs(JA, [x; u], zeros(n+m,1));
B = subs(JB, [x; u], zeros(n+m,1));
d = simplify(f - (A*x + B*u));
