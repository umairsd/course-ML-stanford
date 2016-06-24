function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

%% Cost Function
% X         m x 2
% X'        2 x m
% theta     2 x 1
% t1        m x 1
t1 = X * theta;

% y         m x 1
% t2, t3    m x 1
t2 = t1 - y;
t3 = t2 .* t2;

% Calculate the first term of J
J_noReg = sum(t3) / (2 * m);

% Regularization Parameter
theta_squared = theta .* theta;

% We must ignore theta0, which in this matrix is theta(1)
theta_sum = sum(theta_squared(2:end));

regularization = theta_sum * lambda / (2 * m);

J = J_noReg + regularization;

% =========================================================================



grad = grad(:);

end
