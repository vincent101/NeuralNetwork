function [alpha, beta, testMSE, times] = stopByIncreaseOfTestMSE(trainX, trainY, testX, testY, M, eta)
    % initial alpha and beta be null, run function sNeuralNet
    alpha = [];
    beta = [];
    testMSE = [];
    t = 10000;
    for i = 1:t
        [alpha,beta,r1] = sNeuralNet(trainX, trainY, M, eta, alpha, beta);
        [predictY,r2] = sPredict(testX, testY, alpha, beta); %#ok<*ASGLU>
        testMSE(i) = r2; %#ok<*AGROW>
        if i-1 >=1
            if testMSE(i) >= testMSE(i-1)
                break;
            end
        end
    end
    testMSE = testMSE(length(testMSE));
    times = i;
end
