clear; clc; close all;
%% Problem 1.1
% make 50 points equally spaced between 0 & 1 as the 
% input dataset {u^(i)}^50_{i=1}. 0 is not included in u_i.
u_i = linspace(0, 1, 51)';
u_i = u_i(2:51);

%% Problem 1.2
% Define v^(i) = u^(i)sin(2πu^(i)) + e^(i). e^(i) is the uncertainty 
% drawn from a uniform distribution over the interval [−0.1, 0.1].
rng(0); % rng used just to make sure rng shows the same val 
e_i = unifrnd(-0.1, 0.1, [50, 1]);
v_i = u_i.*sin(2*pi*u_i) + e_i;

%% Problem 1.3 - 1.4
% Regression problem is stated in the work sheet So is the E(θ). 
% Assume that M = 3, find the minimizer!

% Firstly I'll compute minimzer through MATLAB, 
% then I'll present the derivation in the report.

X = [ones(50,1), u_i, u_i.^2, u_i.^3];
theta_minm = (X' * X) \ (X' * v_i);

%% Problem 1.5
% Plot u ↦ h_{θ^∗}(u) in [0,1] domain. Also plot

u_test = linspace(0, 1, 50)';
v_test = u_test.*sin(2*pi*u_test) + unifrnd(-0.1, 0.1, [50, 1]);

% Calculate predicted values using the learned model
h_theta_star = [ones(50, 1), u_test, u_test.^2, u_test.^3]*theta_minm;

% Plot the learned model
figure(1)
plot(u_test, h_theta_star, 'r', 'LineWidth', 1.5);
hold on;
plot(u_i, v_i, 'g', 'LineWidth', 1.5);
plot(u_test, u_test.*sin(2*pi*u_test), 'b', 'LineWidth', 1.5);
legend('Fitted Third Order Model','Original Model','Original Model without uncertainty');
xlabel('u');
ylabel('v');
title('Model Comparison');
grid on;
hold off;

rms_error = sqrt(mean((h_theta_star - v_test).^2));
fprintf('The RMS error is: %.4f\n', rms_error);

%% Problem 1.6
% Repeat previous steps with M = 1 and 9
X_1 = [ones(50,1), u_i];
X_9 = [ones(50,1), u_i, u_i.^2, u_i.^3, u_i.^4, u_i.^5, u_i.^6, u_i.^7, u_i.^8, u_i.^9];
theta_minm_1 = (X_1' * X_1) \ (X_1' * v_i);
theta_minm_9 = (X_9' * X_9) \ (X_9' * v_i);
h_theta_star_1 = [ones(50, 1), u_test]*theta_minm_1;
h_theta_star_9 = [ones(50, 1), u_test, u_test.^2, u_test.^3, u_test.^4, u_test.^5, u_test.^6, u_test.^7, u_test.^8, u_test.^9]*theta_minm_9;


figure(2)
plot(u_test, h_theta_star_1, 'r', 'LineWidth', 1.5);
hold on;
plot(u_i, v_i, 'g', 'LineWidth', 1.5);
plot(u_test, u_test.*sin(2*pi*u_test), 'b', 'LineWidth', 1.5);
legend('Fitted First Order Model','Original Model','Original Model without uncertainty');
xlabel('u');
ylabel('v');
title('Model Comparison');
grid on;
hold off;

rms_error_1 = sqrt(mean((h_theta_star_1 - v_test).^2));
fprintf('The RMS error for first order model is: %.4f\n', rms_error_1);

figure(3)
plot(u_test, h_theta_star_9, 'r', 'LineWidth', 1.5);
hold on;
plot(u_i, v_i, 'g', 'LineWidth', 1.5);
plot(u_test, u_test.*sin(2*pi*u_test), 'b', 'LineWidth', 1.5);
legend('Fitted Ninth Order Model','Original Model','Original Model without uncertainty');
xlabel('u');
ylabel('v');
title('Model Comparison');
grid on;
hold off;

rms_error_9 = sqrt(mean((h_theta_star_9 - v_test).^2));
fprintf('The RMS error for ninth order model is: %.4f\n', rms_error_9);