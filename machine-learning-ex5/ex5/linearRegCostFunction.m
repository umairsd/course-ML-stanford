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

%% Gradient
grad_nonV = zeros(size(theta));

n = size(theta);

for j = 1:n
    grad_j = 0;
    
    for i = 1:m
        x_i = X(i,:);               % 1 x n
        y_i = y(i,:);               % 1 x 1
        % theta                     % n x 1
        h_i = x_i * theta;          % 1 x 1
        
        grad_j = grad_j + (h_i - y_i) * x_i(j);
    end
    
    grad_nonV(j) = grad_j / m;
end

regTerm = (lambda/m) * theta;
regTerm(1) = 0;

grad = grad_nonV + regTerm;
grad = grad(:);

end
