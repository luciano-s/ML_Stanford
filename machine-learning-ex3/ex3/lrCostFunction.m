function [J, grad] = lrCostFunction(theta, X, y, lambda)

m = length(y); 


J = 0;
grad = zeros(size(theta));


h = sigmoid(X * theta);


grad = grad(:);


J = (-y' * log(h) - (1-y)' * log(1-h))/m + (lambda/(2*m))*sum(theta(2:end).^2);
grad = X' * (h - y)/m + (lambda*theta)/m;
grad(1) -= (lambda*theta(1))/m;

end
