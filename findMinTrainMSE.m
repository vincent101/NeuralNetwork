function minTrainMSE = findMinTrainMSE(trainX, trainY, M, eta, times)
    % initial alpha and beta be null, run function sNeuralNet
    minTrainMSE = [];
    for j = 1:times
        alpha = [];
        beta = [];
        trainMSE = [];
        t = 10000;
        for i = 1:t
            [alpha,beta,r1] = sNeuralNetWithoutSeed(trainX, trainY, M, eta, alpha, beta);
            trainMSE(i) = r1; %#ok<*AGROW,*SAGROW>
        end
        minTrainMSE(j) = min(trainMSE);
    end
    minTrainMSE = min(minTrainMSE);
end