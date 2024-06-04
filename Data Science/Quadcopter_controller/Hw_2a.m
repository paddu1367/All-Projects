%loading the data
load dataexp.mat;
%assigning values
t= log(:,1);
u= log(:,2);
x1= log(:,3);
x2= log(:,4);

% Concatanate the states and inputs
xexp=[x1';x2'];
uexp = u';
T = 200;
nstart = 10;
n=2;
P = sdpvar(n);
Q = sdpvar(T, n, 'full');
LMI = [P, dd_vectorize(xexp, nstart+1,T)*Q;
    Q'*dd_vectorize(xexp, nstart+1,T)', P ];

ops = sdpsettings('solver', 'sedumi');
Fset = [P >= 0] + [LMI >= 0] + [dd_vectorize(xexp, nstart, T) *Q - P == 0];
prob = optimize(Fset, [], ops);

Q = value(Q);
P = value (P);
K = dd_hankel(uexp, nstart, 1, T)*Q*inv(P);
clearvars -except K
save K