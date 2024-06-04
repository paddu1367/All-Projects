clear; clc;
% Warning: Since input and initial value is random, sometime design fails.
% If it happens, rerun the code.

% System parameters
n = 2; % System dimension
m = 1; % Number of inputs

M = 2; L = 1; g = 9.81; mu = 0.01; % Mass Length Gravity Friction
delta = 0.1; % Sampling interval in seconds

% Feasibility for Control Design Parameters
nstart = 1;  % Initial time of the data I want to use
T = 10;       % Amount of data we use to design the system
DidConditionsHold = 1;

tspan = 0:delta:2*T*delta;


xamp = 0.01;
xexp = 2*xamp*rand(n, 1) - xamp; % A random initial condition in the range of [-0.01, 0.01]

uamp = 0.2;
uexp = 2*uamp*rand(m, length(tspan)) - uamp; % Random inputs in the range [-0.2, 0.2]

% Simulate the system
for k = 1:length(tspan)-1
    xexp(1, k+1) = xexp(1, k) + delta*xexp(2, k);
    xexp(2, k+1) = delta*g/L*sin(xexp(1, k)) + ...
                   (1 - delta*mu/M/L^2)*xexp(2, k) + ...
                   delta/M/L^2*uexp(k);
    dexp(1, k) = 0;
    dexp(2, k) =  delta*g/L*(sin(xexp(1, k)) - xexp(1, k));
end

subplot(2,1,1)
plot(tspan, xexp(1, :))
ylabel('Position')

subplot(2,1,2)
plot(tspan, xexp(2, :))
ylabel('Velocity')

% Check the assumptions
% Assumption 4
matrix1 = [ dd_hankel(uexp, nstart, 1, T); 
            dd_vectorize(xexp, nstart, T)];

% rank == row?
if (rank(dd_vectorize(xexp, nstart+1, T)) - size(dd_vectorize(xexp, nstart+1, T), 1) == 0)
    fprintf('X_{1, T} is full row rank \n');
else
    fprintf('X_{1, T} is full NOT row rank \n');
    DidConditionsHold = 0;
end

if (rank(matrix1) - size(matrix1, 1) == 0)
    fprintf('[U; X] is full row rank \n');
else
    DidConditionsHold = 0;
    fprintf('[U; X] is full NOT row rank \n');
end

% Assumption 5
FindMinimumGamma; % Computes the best gamma for Equation (52)
matrix2 = dd_vectorize(dexp, nstart, T)*dd_vectorize(dexp, nstart, T)' - ...
          gamma*dd_vectorize(xexp, nstart+1,T)*dd_vectorize(xexp, nstart+1,T)';
alpha = max(roots([1, -2*gamma, -4*gamma]))*1.000001; % Computes the best alpha for given gamma


if  (max(eig(matrix2)) <= 0)
    fprintf('Assumption 5 holds \n');
else
    DidConditionsHold = 0;
    fprintf('Assumption 5 does NOT hold \n');
end


if DidConditionsHold == 0
    fprintf('WARNING: Assumptions does NOT hold! \n');
    return
else
    fprintf('All assumptions hold. \n');
end


% LMI Part
P = sdpvar(n); % This defines nxn symmetric matrix
Q = sdpvar(T, n, 'full'); % This defines Txn full matrix

% Constraints
LMI1_11 = P - alpha*dd_vectorize(xexp, nstart+1, T)*dd_vectorize(xexp, nstart+1, T)';
LMI1_12 = dd_vectorize(xexp, nstart+1, T)*Q;
LMI1_13 = dd_vectorize(xexp, nstart+1, T);
LMI1_22 = P;

LMI1 = [LMI1_11 , LMI1_12;
        LMI1_12', LMI1_22];
LMI2 = [eye(T), Q; Q', P];

Fset = [P >= 0] + [LMI1 >= 0] + ...
       [dd_vectorize(xexp, nstart, T)*Q - P == 0 ] + ...
       [LMI2 >= 0];

prob = optimize(Fset, []);

P = value(P);
Q = value(Q);
LMI1 = value(LMI1);
LMI2 = value(LMI2);
alpha = value(alpha);

if( min(eig(P)) < 0 )
    DidConditionsHold = 0;
    fprintf('P is not Positive Definite \n');
end
if( min(eig(LMI1)) < 0 )
    DidConditionsHold = 0;
    fprintf('LMI1 is not Positive Definite \n');
end
if( min(eig(LMI2)) < 0 )
    DidConditionsHold = 0;
    fprintf('LMI2 is not Positive Definite \n');
end

K = dd_hankel(uexp, nstart, 1, T)*Q*inv(dd_vectorize(xexp, nstart, T)*Q);

if DidConditionsHold == 1
    save('gains.mat', 'K');
end


