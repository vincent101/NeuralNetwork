% load Auto.data and define attribution
[mpg,cylinders,displacement,horsepower,weight,acceleration,year1,origin,name] = loadAuto('Auto.csv');

% create dummy variables for origin
originD = dummyvar(origin);
% normalize the attributions and combine columns
X = [horsepower weight year1];
X = zscore(X);
X = [X originD];
Y = mpg;
% Y = zscore(Y);

% set seed
s = RandStream('mt19937ar','Seed', 0506);
RandStream.setGlobalStream(s);

% devide into train and test equally
[trainX, trainY, testX, testY] = divideInto2Equally(X, Y);

% set the null alpha and beta starting
alpha = [];
beta = [];
M = 3;
eta = 0.01;
% fit neural net regression
[alpha, beta, trainMSE] = sNeuralNet(trainX, trainY, M, eta, alpha, beta);

% predict using fitted neural net regression
[predictY, testMSE] = sPredict(testX, testY, alpha, beta); 

% stop when trainMSE not changed by more than 1%, get the trainMSE= 47.6853
[al1, be1, mse1, times1] = stopByDecreaseRateOfTrainMSE(trainX, trainY, M, eta);

% stop when testMSE increase, get the testMSE = 6.9622
[al2, be2, mse2, times2] = stopByIncreaseOfTestMSE(trainX, trainY, testX, testY, M, eta);

% use k-CrossValiation find the optimal value of M
[aveMSE, bestM, testMSE] = KCrossValidation(X, Y);  %#ok<*ASGLU>
% result: bestM = 6 with the testMSE = 7.2048

% M=3, eta=0.01, t=100, different weight in [-0.7,0.7], without seed
sTestMSE = specialCase(trainX,trainY,testX,testY);
box = boxplot(sTestMSE);

% M=1,2,3, sta=0.1,0.01,0.001, stopTimes= 100,200,300
smallTable = tryDiffValue(trainX, trainY);
csvwrite('smallTable.csv', smallTable);

% M=3, eta=0.1, find the min TrainMSE in 4 times, without seed
minTrainMSE = findMinTrainMSE(trainX, trainY, 3, 0.1, 4);
% minTrainMSE = 6.5491

% fit the linear regression
X = [horsepower weight year1 origin];
Y = mpg;
[trainX, trainY, testX, testY] = divideInto2Equally(X, Y);
lm.fit = fitlm(trainX,trainY);
predictY = predict(lm.fit, testX);
testMSE = sum((testY-predictY).^2)/length(testY);
% testMSE = 10.1359




