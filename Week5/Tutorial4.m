close all;
clear;
clc;
%% Problem 1
% Create a code that will iterate through the dataset (using a while loop).
% Compute the cost for each parameter update and for each time you went
% through the entire dataset. Keep going until a convergence of less than
% 0.0001 has been reached (use the cost over the full dataset for this).

% This assume the independent data is only RnD variable
data = readtable('startup_data.csv'); % use readtable rather than csvread
x_i = data.R_D;
y_i = data.Profit;
x_i_dum = (x_i - mean(x_i)) / std(x_i);
x_i = (x_i - mean(x_i)) / std(x_i);
x_i = [ones(size(x_i)), x_i];
theta = zeros(2, 1);
h = @(theta, x_i) x_i * theta;
C = @(theta, x_i, y_i) 1/(2*length(y_i)) * (h(theta, x_i) - y_i)' ...
* (h(theta, x_i) - y_i);
gradC = @(theta, x_i, y_i) 1/length(y_i) * (x_i' * (h(theta, x_i) - y_i));

% Stochastic gradient descent
alpha = 0.001;  
tol = 1e-4;  % convergence threshold
max_iter = 1e5;  % maximum iterations to prevent infinite loop
iter = 0;
idx = 0;
C_prev = inf;  % Initialize the previous cost to a large number
C_iter_history = zeros(max_iter * length(y_i), 1);  % Cost for each iteration
C_full_history = zeros(max_iter, 1);  % Cost for the full dataset

while true
    % Randomly shuffle the dataset
    rand_indices = randperm(length(y_i));
    x_i = x_i(rand_indices, :);
    y_i = y_i(rand_indices);
    
    for i = 1:length(y_i)
        % Update the parameters for each training example
        theta = theta - alpha * gradC(theta, x_i(i, :), y_i(i));

        C_iter = C(theta, x_i(i, :), y_i(i));
        idx = idx + 1;
        C_iter_history(idx) = C_iter;
    end
    
    % Compute and store the cost for the full dataset
    C_full = C(theta, x_i, y_i);
    iter = iter + 1;
    C_full_history(iter) = C_full;
    disp(['Current iteration: ', num2str(iter)]);
    % Check for convergence
    if abs(C_full - C_prev) < tol
        break;
    elseif iter >= max_iter
        disp('Maximum iterations reached without convergence.')
        break;
    end
    
    % Store the current cost for the next iteration
    C_prev = C_full;
end

% Trim cost histories to the number of iterations actually performed
C_iter_history = C_iter_history(1:idx);
C_full_history = C_full_history(1:iter);

% Plot the cost for each iteration
figure;
plot(C_iter_history);
title('Cost for Each Iteration');
xlabel('Iteration');
ylabel('Cost');

% Plot the cost for the full dataset over iterations
figure;
plot(C_full_history);
title('Cost for Full Dataset Over Iterations');
xlabel('Iteration');
ylabel('Cost');