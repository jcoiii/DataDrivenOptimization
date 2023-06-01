clear all; clc; close all;
%% Problem 2.1
% Re-do Tutorial 2

% Define x_i and y_i
ds = readtable('iris1.xlsx');
x_i = [ds.SepalLengthCm, ds.PetalLengthCm];
y_i = ds.Class;

% Solve minimization with cvx tb
[m,n] = size(x_i);
cvx_begin
    variables w(n) b;
    minimize( norm(w) );
    subject to
        y_i .* (x_i*w + b) >= 1;
cvx_end

% Optimal values
w_0_star = w;
b_0_star = b;

% decision boundary using results w and b
x = linspace(min(x_i(:,1)),max(x_i(:,1)));
f = @(x) -(w_0_star(1)*x + b_0_star)/w_0_star(2);
y = f(x);

% Plot the result
scatter(x_i(y_i == 1,1), x_i(y_i == 1,2),'b+')
hold on
scatter(x_i(y_i == -1,1), x_i(y_i == -1,2), '+r')
plot(x,y,'k')
legend('Setosa','Versicolor','Decision Boundary')
title('Sepal length vs Petal length')
hold off

%% Problem 2.2
% Now shift the input value x_i = x_i + [π π/e]
shifter_val = [pi pi/exp(1)];
x_i_new = x_i  + shifter_val;

[m_new,n_new] = size(x_i_new); % Size is basically the same as the OG data
cvx_begin
    variables w_1(n_new) b_1;
    minimize( norm(w_1) );
    subject to
        y_i .* (x_i_new*w_1 + b_1) >= 1;
cvx_end

% Optimal values
w_1_star = w_1;
b_1_star = b_1;

% decision boundary using results w_1 and b_1
x_new = linspace(min(x_i_new(:,1)),max(x_i_new(:,1)));
f_new = @(x) -(w_1_star(1)*x + b_1_star)/w_1_star(2);
y_new = f_new(x_new);

% Plot the result
figure(2)
scatter(x_i_new(y_i == 1,1), x_i_new(y_i == 1,2),'b+')
hold on
scatter(x_i_new(y_i == -1,1), x_i_new(y_i == -1,2), '+r')
plot(x_new,y_new,'k')
legend('Setosa','Versicolor','Decision Boundary')
title('Sepal length vs Petal length (shifted)')
hold off

%% Problem 2.3
% With the initial x_i, rotate it with π/3 in ccw direction
rot_val = pi/3;
rot_mx = [cos(rot_val) -sin(rot_val); sin(rot_val) cos(rot_val)];
x_i_rot = x_i*rot_mx;

[m_rot,n_rot] = size(x_i_rot);% Size is basically the same as the OG data
cvx_begin
    variables w_2(n_rot) b_2;
    minimize( norm(w_2) );
    subject to
        y_i .* (x_i_rot*w_2 + b_2) >= 1;
cvx_end

% Optimal values
w_2_star = w_2;
b_2_star = b_2;

% decision boundary using results w_2 and b_2
x_new_rot = linspace(min(x_i_rot(:,1)),max(x_i_rot(:,1)));
f_new_rot = @(x) -(w_2_star(1)*x + b_2_star)/w_2_star(2);
y_new_rot = f_new_rot(x_new_rot);

% Plot the result
figure(2)
scatter(x_i_rot(y_i == 1,1), x_i_rot(y_i == 1,2),'b+')
hold on
scatter(x_i_rot(y_i == -1,1), x_i_rot(y_i == -1,2), '+r')
plot(x_new_rot,y_new_rot,'k')
legend('Setosa','Versicolor','Decision Boundary')
title('Sepal length vs Petal length (Rotated)')
hold off

%%
% I made the code based on the solution given in Brightspace, so most of
% the function should be the same as the solution given.