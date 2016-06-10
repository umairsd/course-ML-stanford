function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta); % number of features

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

tempJ = 0;

%% Non-vectorized -- J
for i = 1:m
    % theta                     % (n+1) x 1
    x_i = X(i,:);               % 1 x (n+1)
    y_i = y(i,:);               % 1 x 1
    
    h_i = sigmoid(x_i * theta); % 1 x 1
    
    fst_i = y_i * log(h_i);
    snd_i = (1 - y_i) * log(1 - h_i);
    
    tempJ = tempJ - fst_i - snd_i;
end

J_nonV = tempJ / m;


%% Vectorized -- J
% X is m * (n+1) matrix. Theta is (n+1) x 1 vector
% X * theta                     % m x 1 
h_theta = sigmoid (X * theta);  % m x 1

fst_log = log(h_theta);         % m x 1
fst_log_T = fst_log';           % 1 x m
fst = fst_log_T * y;            % 1 x 1 (1 x m times m x 1)

snd_log = log(1 - h_theta);     % m x 1
snd_log_T = snd_log';           % 1 x m
snd = snd_log_T * (1 - y);      % 1 x 1

J_V = - (fst + snd) / m;


%% Gradient (Non-vectorized)
grad_nonV = zeros(size(theta));

for j = 1:n
    grad_j = 0;
    for i = 1:m
        x_i = X(i,:);               % 1 x (n+1)
        y_i = y(i,:);               % 1 x 1
        h_i = sigmoid(x_i * theta); % 1 x 1
        
        grad_j = grad_j + (h_i - y_i) * x_i(j);
    end
    
    grad_nonV(j) = grad_j / m;
end

%% Gradient (Vectorized)
% X                 % m x (n+1)
% X'                % (n+1) x m
% theta             % (n+1) x 1
% X * theta         % m x 1
% y                 % m x 1
% sigmoid(X*theta)  % m x 1
% => result         % (n+1) x 1 

grad_V_summation = X' * (sigmoid(X * theta) - y);
grad_V = grad_V_summation ./ m;


%% Return 
% Both vectorized and non-vectorized solutions produce identical results
J = J_V;
grad = grad_V;

% =============================================================

end
