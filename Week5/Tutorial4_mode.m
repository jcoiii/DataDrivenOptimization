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
x_i = [ones(size(x_i)), x_i];
theta = zeros(2, 1);
h = @(theta, x_i) x_i * theta;
C = @(theta, x_i, y_i) 1/(2*length(y_i)) * (h(theta, x_i) - y_i)' ...
* (h(theta, x_i) - y_i);
gradC = @(theta, x_i, y_i) 1/length(y_i) * (x_i' * (h(theta, x_i) - y_i));
%%

% SGD
alpha = 0.0001;  
tol = 1e-4;  % convergence threshold
max_iter = 1e5;  % maximum iter to prevent infinite loop
iter = 0;
idx = 0;
C_prev = 0;  % define the previous cost to a large number
C_iter_hist = zeros(max_iter * length(y_i), 1);  % each iteration cost dummy
C_full_hist = zeros(max_iter, 1);  % full set cost dummy

h_i = 0;

while true
    for hl = h_i+1:h_i+100
    data_num = h_i+1;
    data_num_lim = h_i+100;
    x_i_n = x_i((data_num:data_num_lim),:);
    y_i_n = y_i((data_num:data_num_lim));

    rand_indices = randperm(length(y_i_n));
    x_i_n = x_i_n(rand_indices, :);
    y_i_n = y_i_n(rand_indices);
        
        for i = 1:length(y_i_n)
            theta = theta - alpha * gradC(theta, x_i_n(i, :), y_i_n(i));
    
            C_iter = C(theta, x_i_n(i, :), y_i_n(i));
            idx = idx + 1;
            C_iter_hist(idx) = C_iter;
        end
        
    
        C_full = C(theta, x_i_n, y_i_n);
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
    
    if C_prev < tol
       break;
    elseif iter >= max_iter
       disp('lol, such lot iteration but no convergence.')
       break;
    end
    end
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
