%[J grad] = nnCostFunction(nn_params, ...
%                                   input_layer_size, ...
%                                   hidden_layer_size, ...
%                                   num_labels, ...
%                                   X, y, lambda)

Theta1_ = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2_ = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));


m = size(X, 1);
         

J = 0;
Theta1_grad = zeros(size(Theta1_));
Theta2_grad = zeros(size(Theta2_));



X_ = [ones(m, 1) X];

a2 = sigmoid(Theta1_*X_')';
a2 = [ones(m, 1) a2];
a3 = sigmoid(Theta2*a2')';
y_ = zeros(m,num_labels);

for (c =1:1:m)
 y_(c,y(c)) = 1;  
end

J = (1/m)*(-y_.*log(a3) - (1-y_).*log(1-a3));
J = sum(sum(J')');

reg = (lambda/(2*m))*(sum(sum(Theta1_(:,2:end)'.^2))  + sum(sum(Theta2_(:,2:end)'.^2)) );
J = J+reg;
%t=1
for(t = 1:1:m )
  yt = y_(t,:);
  a1 = X(t,:);
  a1 = [1, a1];
  z2 = Theta1*a1';
  a2 = sigmoid(z2)';
  a2 = [1, a2];
  z3 = Theta2*a2';
  a3 = sigmoid(z3)';
  
  delta3 = a3-yt;
  delta2 = (Theta2_)'*delta3'.*sigmoidGradient([1;z2]);
  delta2 = delta2(2:end);
  
  Theta1_grad = Theta1_grad + delta2*a1;
  Theta2_grad = Theta2_grad + delta3'*a2;
  
end