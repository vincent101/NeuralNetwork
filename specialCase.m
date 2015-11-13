function testMSE = specialCase(trainX,trainY,testX,testY)
    % fix M=3, eta=0.01  with testMSE increase stopping condition0
    % times = 100
    M = 3;
    eta = 0.01;
    testMSE = [];
    t = 100;
    for i = 1:t
        % initial alpha and beta be null, run function sNeuralNet
        alpha = []; %#ok<NASGU>
        beta = []; %#ok<NASGU>
        [al2, be2, mse2, times2] = stopByIncreaseOfTestMSEWithoutSeed(trainX, trainY, testX, testY, M, eta); %#ok<*ASGLU>
        testMSE(i) = mse2; %#ok<*AGROW>
    end
end
