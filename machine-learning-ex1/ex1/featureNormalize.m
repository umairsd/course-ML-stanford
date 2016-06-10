function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%       

mu = mean(X);
sigma = std(X);

newX = zeros(size(X,1), size(X,2));

% This is a clunky way to calculate X_norm, but I kept running
% into errors when attempting this, so that's why I am using 
% two loops to be explicit about what I am calculating, and how.
for row = 1:size(X,1)
    for col = 1 : size(X,2)
        numerator = X(row,col) - mu(col);
        newVal = numerator / sigma(col);
        
        newX(row,col) = newVal;
    end
    
end

X_norm = newX;

end
