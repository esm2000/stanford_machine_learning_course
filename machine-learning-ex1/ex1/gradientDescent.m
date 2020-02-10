function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    % set the number of training examples
    m = length(y);

    % make predictions based on the parameters and data 
    hypothesis = X * theta;

    % calculate the error 
    error = hypothesis - y;

    % update theta
    theta(1,1) = theta(1,1) - (alpha * ( 1 / m ) * sum( error ))

    theta(2,1) = theta(2,1) - (alpha * ( 1 / m ) * sum( error' * X(:,2)))


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
