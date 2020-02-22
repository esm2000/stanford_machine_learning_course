function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
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


% zero

theta_zero = theta(1,1);
X_zero = X(:, 1);

% rest

theta_rest = theta(2:end, 1);
X_rest = X(:, 2:end)

% calculate the hypotheses

hypothesis = sigmoid( X * theta );

% calculate cost
 
J = ( 1 / m ) * sum( ( -y .* log( hypothesis ) ) - ( ( 1 - y ) .* log( 1 - hypothesis )) )
J = J + ( ( lambda / ( 2 * m ) ) * sum( theta_rest .^ 2 )  ) 

grad_zero = ( 1 / m ) * sum( (hypothesis - y) .* X_zero );
grad_rest = ( ( 1 / m ) * sum( ( hypothesis - y ) .* X_rest ) )' + ( ( lambda / m ) .* theta_rest );

grad = [ grad_zero; grad_rest ];


% =============================================================

end
