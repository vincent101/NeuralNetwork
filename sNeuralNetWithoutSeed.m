function [alpha, beta, trainMSE] = sNeuralNetWithoutSeed(trainX, trainY, M, eta, alpha, beta)
    [N,p] = size(trainX);
    [N1,K] = size(trainY);
    if N ~= N1
        % check the demension of trainX and trainY
        disp('Wrong dimensions of trainX and trainY!')
    end
    if K ~= 1 
        % check if K =1
        disp('Wrong K, the columns of trainY!')
    end
    if isempty(alpha) || isempty(beta)
        alpha = 1.4*rand(M,p+1)-0.7;
        beta = 1.4*rand(1,M+1)-0.7;
    end
    z = zeros(N,M);
    y = zeros(N,1);
    delta = zeros(1,N);
    s = zeros(M,N);

    for i = 1:N
        x = trainX(i,:);
        for m = 1:M
            z(i,m) = sigmoid(alpha(m,1)+alpha(m,2:p+1)*x');
        end
%         y(i) = sigmoid(beta(1)+beta(2:M+1)*z(i,:)');
        y(i) = beta(1)+beta(2:M+1)*z(i,:)';
        delta(i) = 2*(trainY(i)-y(i));
        beta(1) = beta(1) + eta/N * delta(i);
        for m = 1:M
            s(m,i) = beta(m)*delta(i)*dsigmoid(alpha(m,1)+alpha(m,2:p+1)*x');
            alpha(m,1) = alpha(m,1) + eta/N * s(m,i);
        end
    end

    trainMSE = sum((y-trainY).^2)/N;
    beta(2:M+1) = beta(2:M+1) + eta/N * (delta * z);
    alpha(:,2:p+1) = alpha(:,2:p+1) + eta/N * (s * trainX);
    
end


