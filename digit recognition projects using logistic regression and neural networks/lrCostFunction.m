function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples


J = 0;
grad = zeros(size(theta));




z=sigmoid(X*theta);


J = (1/m)*(-y'* log(z) - (1 - y)'*log(1-z))+(lambda/(2*m))*theta(2:end)'*theta(2:end);

grad(1)=sum(X(:,1)'*(z-y))/m;
grad(2:end)=(X(:,2:end)'*(z-y)/m)+((theta(2:end)).*lambda/m);





% =============================================================

grad = grad(:);

end
