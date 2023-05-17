
%% Pre coding Problem 1
% GD and SGD differ in how they update the parameters:

% GD computes the gradient of the cost function with respect to the 
% parameters for the entire training dataset. It updates the parameters 
% in the direction of the negative gradient to minimize the cost function. 
% Basically it uses all of the data at each iteration.

% SGD updates the parameters for each training example one at a time. 
% it computes the gradient of the cost function and updates the parameters 
% for each individual data point. 

% Additional notes*:
% SGD uses one data point at a time, the path to the minimum of the cost 
% function can be noisier than the path taken by GD.

% Read about Mini-Batch Gradient Descent*. <- comes in between GD and SGD 

% So, the main difference of GD and SGD is how they use the data to compute 
% the gradient and update the parameters. GD uses all the data at once, 
% while SGD uses one data point at a time.

%% Pre coding Problem 2
% The update is performed for each training data i,

%% Pre coding Problem 3
% The convergence of the cost function of SGD is more noisy and less smooth
% than in GD. This is because the parameter updates in SGD are based on 
% individual training data i, which may lead to frequent changes in 
% direction and magnitude of the updates.

% SGD should converges faster than Batch Gradient Descent in terms of 
% computation time, in the case of large datasets, 
% because it makes updates with every data i as opposed to waiting to 
% see the entire dataset.


%%
close all;
clear;
clc;
%% Problem 1
% Create a code that will iterate through the dataset (using a while loop).
% Compute the cost for each parameter update and for each time you went
% through the entire dataset. Keep going until a convergence of less than
% 0.0001 has been reached (use the cost over the full dataset for this).

% This assume the independent data is only RnD variable
data = normalize(readtable('startup_data.csv')); % readtable ~ csvread
x_i = data.R_D;
y_i = data.Profit;
x_i_dum = x_i;
x_i = (x_i - mean(x_i)) / std(x_i);
x_i = [ones(size(x_i)), x_i];
theta = zeros(2, 1);
h = @(theta, x_i) x_i * theta;
C = @(theta, x_i, y_i) 1/(2*length(y_i)) * (h(theta, x_i) - y_i)' ...
* (h(theta, x_i) - y_i);
gradC = @(theta, x_i, y_i) 1/length(y_i) * (x_i' * (h(theta, x_i) - y_i));
%%

% SGD
alpha = 0.01;  
tol = 1e-4;  % convergence threshold
max_iter = 3e3;  % maximum iter to prevent infinite loop
iter = 0;
idx = 0;
C_prev = inf;  % define the previous cost to a large number
C_iter_hist = zeros(max_iter * length(y_i), 1);  % each iteration cost
C_full_hist = zeros(max_iter, 1);  % full dataset cost

h_i = 0;

while true
    for h = (h_i+1):(h_i+100)
        rand_indices = randperm(length(y_i));
        x_i = x_i(rand_indices, :);
        y_i = y_i(rand_indices);
        
        for i = 1:length(y_i)
            theta = theta - alpha * gradC(theta, x_i(i, :), y_i(i));
    
            C_iter = C(theta, x_i(i, :), y_i(i));
            idx = idx + 1;
            C_iter_hist(idx) = C_iter;
        end
        
    
        C_full = C(theta, x_i, y_i);
        iter = iter + 1;
        C_full_hist(iter) = C_full;
        disp(['n-iteration now is ', num2str(iter)]);
    
        % if abs(C_full - C_prev) < tol
        %     break;
        % elseif iter >= max_iter
        %     disp('Max iterations achieved but no convergence happens.')
        %     break;
        % end
        C_prev = C_full;
    end
    if C_prev < tol
       break;
    elseif iter >= max_iter
       disp('lol, you achieved maximum iter but no convergence.')
       break;
    end
    C_prev = C_full;

end

C_iter_hist = C_iter_hist(1:idx);
C_full_hist = C_full_hist(1:iter);

%% Plot: (1) the cost of each iteration, (2) the cost of the full dataset
figure(1);
plot(C_iter_hist);
title('Cost for Each Iteration');
xlabel('Iteration');
ylabel('Cost');

figure(2);
plot(C_full_hist);
title('Cost for Full Dataset Over Iterations');
xlabel('Iteration');
ylabel('Cost');
