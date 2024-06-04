%loading the data
load dataexp.mat;
%assigning values
t= log(:,1);
u= log(:,2);
x1= log(:,3);
x2= log(:,4);

% Concatanate the states and inputs
x=[x1';x2'];
u = u';
T = 200;
nstart = 5;
%Modelling the data
M = [dd_hankel(u, nstart, 1, T); dd_vectorize(x, nstart, T)]; 
model = dd_vectorize(x, nstart+1, T) *pinv(M);
clearvars -except model
save model
