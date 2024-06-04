function [Z] = dd_vectorize(z, i, N)
Z = z(:, i:(i+N-1));
end

