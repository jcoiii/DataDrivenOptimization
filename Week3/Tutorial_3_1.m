clear all; clc;
%% Task 1
% Load the Iris dataset into Matlab.
ds = readtable('iris.xlsx');

%% Task 2
% Define the independent (input) and dependent (output) variables and store
% them under 𝑥𝑖 and 𝑦𝑖.

x_i = [ds.SepalLengthCm, ds.PetalLengthCm];
y_i = ds.Class;

%% Task 3 
% Visualize the data by making a scatter plot (put sepal length on the x-axis
% and petal length on the y-axis).

scatter(x_i(:,1), x_i(:,2), [], y_i, 'filled');
xlabel('Sepal Length (cm)');
ylabel('Petal Length (cm)');

%% Task 4
% This is where we'll construct the optimization problem, and hence need
% the CVX toolbox. Go back to the lecture slides and find which optimization
% problem we have to solve. During the tutorial, it will be shown how to use
% the CVX toolbox and calculate the optimal values of 𝑤 and 𝑏. Solve the
% optimization problem and report the optimizers.
% Look Slide 12/29 - Lecture 2
cvx_begin
    variables w(2) b;
    minimize( norm(w) );
    subject to
%        y_i * (w'*x_i + b) >= 1;
        y_i .* (x_i*w + b) >= 1;
cvx_end

%% Task 5
% Include the decision boundary in the scatter plot.
x = min(x_i(:,1)):0.001:max(x_i(:,1));
y = (-b - w(1)*x) / w(2);

hold on;
plot(x, y);

%% Task 6
% Calculate the width of the margin (see the slides), and plot the support
% vector lines.
