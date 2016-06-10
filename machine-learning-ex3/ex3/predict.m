function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%


% Add ones to the X data matrix
X = [ones(m, 1) X];


% z(2) = Theta1 * a(1)
% X             5000 x 401
% Theta1        25 x 401
% z(2)          25 x 5000
z_2 = Theta1 * X';
a_2 = sigmoid(z_2);

% Add bias unit to a(2)
a_2 = [ones(1, size(a_2, 2)); a_2];

% a_2           26 x 5000
% Theta2        10 x 26
% a_3           10 x 5000
z_3 = Theta2 * a_2;
a_3 = sigmoid(z_3);

% h_theta       10 x 5000
% This means that we need to use h_theta' so that we can get
% the highest probability for each of the K (K=10) classifers
h_theta = a_3;

[maxVal, classiferIndex] = max(h_theta', [], 2);
p = classiferIndex;


% =========================================================================


end
