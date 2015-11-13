function [alpha, beta, trainMSE, times] = stopByDecreaseRateOfTrainMSE(trainX, trainY, M, eta)
    % initial alpha and beta be null, run function sNeuralNet
    alpha = [];
    beta = [];
    trainMSE = [];
    t = 10000;
    countForStop = 0;
    for i = 1:t
        [alpha,beta,r1] = sNeuralNet(trainX, trainY, M, eta, alpha, beta);
        trainMSE(i) = r1; %#ok<*AGROW,*SAGROW>
        if i-1 >= 1
            if trainMSE(i-1)-trainMSE(i)<1/100*trainMSE(1)
                countForStop = countForStop + 1; %#ok<*REDEF>
               if countForStop == 10
                  break;
               end
            end
        end
    end
    trainMSE = trainMSE(length(trainMSE));
    times = i;
end