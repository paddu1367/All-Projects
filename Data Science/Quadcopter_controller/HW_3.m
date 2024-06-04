%defining model from paper
A=[1.178, 0.001, 0.511, -0.403; 
-0.051, 0.661, -0.011, 0.061;
0.076, 0.335, 0.560, 0.382;
0, 0.335, 0.089, 0.849;];

B=[0.004, -0.087;
   0.467, 0.001;
   0.213, -0.235;
   0.213, -0.016];
nstart = 10;
n=4;
m=2;
tspan=0:0.1:10;
%Defining inputs
u1=0.5*sin(2*pi*0.5*tspan)+0.7*sin(2*pi*0.25*tspan)+0.1*sin(2*pi*0.7*tspan);
u2=0.5*cos(2*pi*0.5*tspan)+0.7*cos(2*pi*0.25*tspan);

xsim = [0; 0; 0; 0;];

%stimulation of system
for k = 2:length(tspan)
    xsim(:,k) =A*[xsim(:, k-1)]+B*[u1(k);u2(k)];

end


%lets try to design a closed loop controller K
T=50;
P = sdpvar(n);
Q = sdpvar(T, n, 'full');
LMI = [P, dd_vectorize(xsim, nstart+1,T)*Q;
    Q'*dd_vectorize(xsim, nstart+1,T)', P ];

ops = sdpsettings('solver', 'sedumi');
Fset = [P >= 0] + [LMI >= 0] + [dd_vectorize(xsim, nstart, T) *Q - P == 0];
prob = optimize(Fset, [], ops);

Q = value(Q);
P = value (P);
K = dd_hankel([u1;u2], nstart, 1, T)*Q*inv(P);

%stimulation for closed loop control system

xsta = [0; 0; 0; 0;];
r=1;
%stimulation of system
for k = 2:length(tspan)
    xsta(:,k) =A*[xsta(:, k-1)-r]+B*K*[xsta(:, k-1)];
end


x1sim = xsta(1, :);
x2sim = xsta(2, :);
x3sim = xsta(3, :);
x4sim = xsta(4, :);
cla; clf;
%plot of simulated results
subplot(2,2,1)
plot(tspan, x1sim);
title('plot of x1');

subplot(2,2,2)
plot(tspan, x2sim);
title('plot of x2');

subplot(2,2,3)
plot(tspan, x3sim);
title('plot of x3');

subplot(2,2,4)
plot(tspan, x4sim);
title('plot of x4');
