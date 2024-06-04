% A bisection algorithm to find minimum gamma that satisfies Equation (52)
tol = 1e-12;
gammau = 50; gammal = 0;

while (gammau - gammal >= tol)
    gamma = (gammau + gammal)/2;
    matrix2 = dd_vectorize(dexp, nstart, T)*dd_vectorize(dexp, nstart, T)' - ...
          gamma*dd_vectorize(xexp, nstart+1,T)*dd_vectorize(xexp, nstart+1,T)';

    if  (max(eig(matrix2)) <= 0)    
        gammau = gamma;
    else
        gammal = gamma;
    end
end
gamma = gammau;