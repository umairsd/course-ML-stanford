function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


%% Part-1: Forward Propagation to Calculate Cost



% X         m x n (e.g. 5000 x 400)
% y         m x 1 (e.g. 5000 x 1)
% Adding a column of ones
X = [ones(m, 1) X];

% % Get the columns (i.e. features *per* example) -- 401
% n = size(X, 2);

% For each example in the training set, let's calculate the h_theta
for i = 1 : m
    % x_i       401 x 1     [(n+1) x 1]
    x_i = X(i, :)';
    
    % First Layer
    % Theta1    25 x 401    [hidden_layer_size x n]
    % z_2       25 x 1
    z_2 = Theta1 * x_i;
    % a_2       25 x 1
    a_2 = sigmoid(z_2);
    
    % Add the bias unit to a_2
    % a_2       26 x 1
    numColsA_2 = size(a_2,2);
    a_2 = [ones(1,numColsA_2); a_2];
    
    
    % Theta2    10 x 26     [num_labels x (hidden_layer_size+1)]
    % z_3       10 x 1
    z_3 = Theta2 * a_2;
    a_2 = sigmoid(z_3);
    
    
    h_theta_i = a_2;
  
    %{
    Also, recall that whereas the original labels (in the variable y) 
    were 1, 2, ..., 10, for the purpose of training a neural network, we 
    need to recode the labels as vectors containing only values 0 or 1 
    %}
    % y_i       10 x 1      [K x 1]
    y_i = zeros(num_labels, 1);
    y_index = y(i);
    y_i(y_index) = 1;    
    
    % Now,
    % h_theta_i     10x1
    % y_i           10x1
     fst = y_i .* log(h_theta_i);    
     snd = (1 - y_i) .* log(1 - h_theta_i);
     
     inner_sum = sum(fst + snd);
     J = J + inner_sum;
end

% Need to divide J by m
J = J / m;
J = -J;



%% Regularized Cost Function


%{
Both Theta1 and Theta2 contains bias units. In our example, Theta1 is 
25x401 and Theta2 is 10x26 matrices. We need to remove the first column
from both of these, for the regularization
%}
% Theta1_reg    25 x 400
Theta1_reg = Theta1(:,2:end);
% Theta2_reg    10 x 25
Theta2_reg = Theta2(:,2:end);

theta1_squared = Theta1_reg .* Theta1_reg;
theta2_squared = Theta2_reg .* Theta2_reg;

total = sum(theta1_squared(:)) + sum(theta2_squared(:));

regularization = lambda * total / (2 * m);

J = J + regularization;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
