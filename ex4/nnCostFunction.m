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
yvector = zeros(num_labels);
yvector(1:num_labels+1:end) = 1; %column or row 1-10 represents values of y(1-10) respectively
Theta1_refmat = ones(size(Theta1));
Theta1_refmat(:,1) = 0;
Theta2_refmat = ones(size(Theta2));
Theta2_refmat(:,1) = 0;

for i = 1:m
  a1 = [1; X(i,:)'];
  z2 = Theta1*a1;
  a2 = [1; sigmoid(z2)];
  z3 = Theta2*a2;
  a3 = sigmoid(z3);
  J = J - yvector(y(i),:)*log(a3) - (1-yvector(y(i),:))*log(1-a3);
end
J = J/m;
J = J + lambda/2/m*(sum(sum((Theta1_refmat.*Theta1).^2)) + sum(sum((Theta2_refmat.*Theta2).^2))); % add regularization terms

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
delta1 = zeros(hidden_layer_size,input_layer_size+1);
delta2 = zeros(num_labels, hidden_layer_size+1);

for i = 1:m
  a1 = [1; X(i,:)'];
  z2 = Theta1*a1;
  a2 = [1; sigmoid(z2)];
  z3 = Theta2*a2;
  a3 = sigmoid(z3);
  
  yita3 = a3 - yvector(y(i),:)';
  yita2 = Theta2'*yita3.*a2.*(1-a2);

  delta1 = delta1 + yita2(2:end)*a1';
  delta2 = delta2 + yita3*a2';  
end
Theta1_grad = (delta1 + lambda*Theta1_refmat.*Theta1)/m;
Theta2_grad = (delta2 + lambda*Theta2_refmat.*Theta2)/m;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
