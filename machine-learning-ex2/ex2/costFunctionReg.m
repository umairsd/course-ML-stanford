function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


%% Calculate J & gradient (non-regularized)
[J_orig, gradient_orig] = costFunction(theta, X, y);

%% Calculate the regularization factors
% NOTE: Theta1 does not need to be regularized
theta_without_theta1 = theta(2 : size(theta, 1));
regJ = lambda * sum(theta_without_theta1 .* theta_without_theta1) / (2 * m);

regG = (lambda/m) .* theta;
regG(1) = 0;

J = J_orig + regJ;
grad = gradient_orig + regG;

% =============================================================

end
