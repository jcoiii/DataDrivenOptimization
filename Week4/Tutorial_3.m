close all;
clear;
clc;
%% Problem 1
% Load and normalize the data, set your 𝑥𝑖 and 𝑦𝑖 variables. We ll use the
% RD spent as our independent variable and the profit as our dependent
% variable.
data = readtable('startup_data.csv'); %use readtable rather than csvread
x_i = data.R_D;
y_i = data.Profit;
x_dum = (x_i - mean(x_i)) / std(x_i);
x_i = (x_i - mean(x_i)) / std(x_i);

%% Problem 2
% Visualize the data by creating a scatter plot.
figure(1)
scatter(x_i, y_i);
xlabel('R&D Spend');
ylabel('Profit');
%% Problem 3
% Define the functions you’ll need for your update step. You can use 
% function handles for this, which you can later call for 
% the given variables.

x_i = [ones(size(x_i)), x_i];

h = @(theta, x_i) x_i * theta;
C = @(theta, x_i, y_i) 1/(2*length(y_i)) * sum((h(theta, x_i) - y_i).^2);
gradC = @(theta, x_i, y_i) 1/length(y_i) * sum(x_i' * (h(theta, x_i) - y_i));
update = @(theta, alpha, gradC) theta - alpha * gradC;

%% Problem 4
% Create the update step with a for loop that runs for a fixed amount of
% iterations.

% initial var 
theta = zeros(2, 1);
alpha = 0.001;
n_i = 100000;
c_val = zeros(n_i, 1);
for i = 1:n_i
    theta = update(theta, alpha, gradC(theta, x_i, y_i));
    c_val(i) = C(theta,x_i,y_i);
end

%% Problem 5
% Create 2 plots:
% (a) Plot ℎ(𝑥, 𝜃) in the scatter plot.
h_val = h(theta, x_i);
figure(2)
scatter(x_dum, y_i);
xlabel('R&D Spend');
ylabel('Profit');
hold on;
plot(x_dum, h_val, 'r');
% hold off;

% (b) Plot the evolution of the cost function over iterations.
figure(3)
plot(1:n_i, c_val, 'b');
xlabel('n_i');
ylabel('Cost C');
title('Evolution of the cost function over iterations');