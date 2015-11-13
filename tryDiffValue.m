function smallTable = tryDiffValue(trainX, trainY)
    smallTable = {};
    M = [1,2,3]; %#ok<*NASGU>
    eta = [0.1, 0.01, 0.001];
    stopTime = [100, 200, 300];
    count = 0;
    for m = M
        for e = eta
            for st = stopTime
                alpha = [];
                beta = [];
                for i = 1:st 
                    [alpha, beta, r1] = sNeuralNetWithoutSeed(trainX, trainY, m, e, alpha, beta);    
                end
                count = count + 1;
                smallTable{count,4} = r1;  %#ok<*AGROW>
                smallTable{count,3} = st;
                smallTable{count,2} = e;
                smallTable{count,1} = m;
            end
        end
    end
end

