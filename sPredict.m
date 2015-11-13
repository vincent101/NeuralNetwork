function [predictY, testMSE] = sPredict(testX, testY, alpha, beta)
    [N,p] = size(testX);
    [M,miss] = size(alpha);
    z = [];
    predictY = [];
%     [1,M+1] = size(beta);
%     z = alpha(:,1) + sum(alpha(:,2:p+1).*testX,2);
    for i = 1:N
        x = testX(i,:);
        for m = 1:M
            z(i,m) = sigmoid(alpha(m,1)+alpha(m,2:p+1)*x');
        end
%         predictY(i) = sigmoid(beta(1)+beta(2:M+1)*z(i,:)');  %#ok<*AGROW>
        predictY(i) = beta(1)+beta(2:M+1)*z(i,:)';  %#ok<*AGROW>
    end
    predictY = predictY';
    testMSE = sum((testY-predictY).^2)/N;
end