function [aveMSE, bestM, testMSE] = KCrossValidation(X, Y)
    % define reasonable eta
    eta = 0.01;
    s = RandStream('mt19937ar','Seed', 0506);
    RandStream.setGlobalStream(s);
    ri = randperm(length(X)); 
    aveMSE = [];
    minMSE = Inf;
    for M = 1:10
        tMSE = [];
        for i = 1:4
            test = ri((i-1)*49+1:i*49);
            train = setdiff(ri, test);
            trainX = X(train,:); %#ok<*NASGU>
            trainY = Y(train,:);
            testX = X(test,:);
            testY = Y(test,:);
            [al, be, mse, times] = stopByIncreaseOfTestMSE(trainX,trainY,testX,testY,M,eta); %#ok<*ASGLU>
            tMSE(i) = mse; %#ok<*AGROW>
        end
        aveMSE(M) = sum(tMSE)/length(tMSE);
        if aveMSE(M)<minMSE
            minMSE = aveMSE(M);
            bestM = M;
        end
    end
    
    train = ri(1:196);
    test = ri(197:length(X));
    trainX = X(train,:);
    trainY = Y(train,:);
    testX = X(test,:);
    testY = Y(test,:);
    [alpha, beta, testMSE, times] = stopByIncreaseOfTestMSE(trainX,trainY,testX,testY,bestM,eta);
    
end