# CS5100 Assignment1 Report
## Neural Network
----
1. Neural Network for regression,   
    to fit NeuralNetwork Regession which all parameters is not fixed,     
    `[alpha, beta, trainMSE] = sNeuralNet(trainX, trainY, M, eta, alpha, `beta   ; 	      
    to predict the test label with fitted alpha, beta from sNeuralNet()  
    `[predictY, testMSE] = sPredict(testX, testY, alpha, beta);`    
  
2. Apply Auto Data set to predict *mpg* given *horsepower*, *weight*, *year* and  *origin*  
    to load the Auto Data set, which has been pre-operated to csv make it easy to load  
    `[mpg,cylinders,displacement,horsepower,weight,acceleration,year1,origin,name] = loadAuto('Auto.csv');`   
    to divid the Data set into attributions and label with Normalize and Dummy.   
    `X = [horsepower weight year1];`  
    `X = zscore(X);`  
    `originD = dummyvar(origin);`     
    `X = [X originD];`    
    `Y = mpg;`    
  
3. Split the Data set randomly into two parts, which serve as training set and test set.    
    `[trainX, trainY, testX, testY] = divideInto2Equally(X, Y);`  
  
4. Try different value of M, eta, number of training step,      
    `smallTable = tryDiffValue(trainX, trainY);`     
    get the *smallTable.csv*, please check it in the folder.
   
5. Try different stopping rules,    
    Stop trainingMSE does not change by more than 1%,     
    `[al1, be1, mse1, times1] = stopByDecreaseRateOfTrainMSE(trainX, trainY, M, eta);`    
    which get the trainMSE = __47.6853__ with M=3, eta=0.01,  
    Stop testMSE increse,     
    `[al2, be2, mse2, times2] = stopByIncreaseOfTestMSE(trainX, trainY, testX, testY, M, eta);`   
    which get the testMSE = __6.9622__ with M=3, eta=0.01,  
  
6. Using 4 Fold-Cross-Validation to find the best value of M,    
    `[aveMSE, bestM, testMSE] = KCrossValidation(X, Y);`  
    which get the bestM = __6__ with eta=0.01,   
7. Train linear regression on training set for predicting *mpg*     
    `X = [horsepower weight year1 origin];`       
    `Y = mpg;`        
    `[trainX, trainY, testX, testY] = divideInto2Equally(X, Y); `             
    `lm.fit = fitlm(trainX,trainY);`      
    `predictY = predict(lm.fit, testX);`      
    `testMSE = sum((testY-predictY).^2)/length(testY);`          
    which get the testMSE = __10.1359__   
    In my opion, NeuralNet algorithm perform better than LinearRegression in Auto Data set.   
  
8. fit NeuralNet for fixed M=3, eta=0.01 with random initial weight in 100 times,       
    `sTestMSE = specialCase(trainX,trainY,testX,testY);`      
    `box = boxplot(sTestMSE)`     
    which get the testMSE in boxplot *Fig_Q8.jpg*, please check it in the folder.
    ![BoxPlot](./F_Q8.jpg)	
  
9. Redo and Train 4 times then use back-propagation,    
    `minTrainMSE = findMinTrainMSE(trainX, trainY, 3, 0.1, 4);`   
    which get the best trainingMSE = __6.5491__   

