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

%forward propagation


%input layer
a1=[ones(m,1) X];

%hidden layer
z2=a1*Theta1';
a2=[ones(m,1) sigmoid(z2)];

%output layer
z3=a2*Theta2';
a3=sigmoid(z3);



%expanding y(putting 1 in the idex value and 0's elsewhere)
eye_matrix = eye(num_labels);
y_matrix = eye_matrix(y,:);

%cost function
J=sum(sum(-y_matrix.*log(a3)-(1-y_matrix).*log(1-a3)))/m+(lambda/(2*m))*(sum(sum(Theta1(:,2:end).^2))+sum(sum(Theta2(:,2:end).^2)));


%back propagation
d3=(a3)-(y_matrix);
d2=(d3*Theta2).*sigmoidGradient([ones(m,1) z2]);
d2=d2(:,2:end);

%gradient
%Theta1_grad(:,1)=(d2'*a1)./m;
%Theta2_grad(:,1)=(d3'*a2)./m;

%regularised gradient
Theta1_grad=(d2'*a1)./m + (lambda/m)*[zeros(size(Theta1,1),1) Theta1(:,2:end)];
Theta2_grad=(d3'*a2)./m + (lambda/m)*[zeros(size(Theta2,1),1) Theta2(:,2:end)];

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
