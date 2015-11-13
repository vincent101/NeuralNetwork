function sig = dsigmoid(x)
    sig = exp(x) / (exp(x)+1)^2;
end