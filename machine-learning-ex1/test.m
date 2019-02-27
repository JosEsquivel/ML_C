m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    h = X*theta;    
    theta = theta - (alpha/m)*(h-y)'*X;
  
    J_history(iter) = computeCost(X, y, theta);

end