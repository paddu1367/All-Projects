function [Z] = dd_hankel(z, i, t, N)
Z = [];
% t counts the number of block rows
for cnt = 1:t
    Z = cat(1, Z, dd_vectorize(z, i + cnt - 1, N));
end
end

