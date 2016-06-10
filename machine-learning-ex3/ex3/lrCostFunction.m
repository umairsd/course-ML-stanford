function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%


%% Cost Function (Vectorized)
% y             m x 1
% X             m x n
% theta         n x 1

% h_theta       m x 1
h_theta = sigmoid(X * theta);

fst = y .* log(h_theta);
snd = (1 - y) .* log (1 - h_theta);

summation = sum(fst + snd);
J_orig = -summation / m;


%% Gradient (Vectorized)
% X                 % m x n
% X'                % n x m
% theta             % n x 1
% X * theta         % m x 1
% y                 % m x 1
% sigmoid(X*theta)  % m x 1
% => result         % n x 1 

grad_summation = X' * (sigmoid(X * theta) - y);
gradient_orig = grad_summation ./ m;


%% Calculate the regularization factors
% NOTE: Theta1 does not need to be regularized
theta_without_theta1 = theta(2 : size(theta, 1));
regJ = lambda * sum(theta_without_theta1 .* theta_without_theta1) / (2 * m);

regG = (lambda/m) .* theta;
regG(1) = 0;

J = J_orig + regJ;
grad = gradient_orig + regG;


% =============================================================

grad = grad(:);

end
