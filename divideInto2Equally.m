function [trainX, trainY, testX, testY] = divideInto2Equally(X, Y)
    % devide into train and test equally
    s = RandStream('mt19937ar','Seed', 0506);
    RandStream.setGlobalStream(s);
    ri = randperm(length(X)); 
    train = ri(1:length(X)/2);
    test = ri(length(X)/2:length(X));
    trainX = X(train,:);
    trainY = Y(train,:);
    testX = X(test,:);
    testY = Y(test,:);
end