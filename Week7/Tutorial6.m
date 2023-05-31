clc; clear;
%% Tutorial 6
% In this tutorial, we’ll construct a neural network which is learning a simple
% function from the corrupted data.
% Steps:
%%
% 1. Create an equally spaced (with step size 0.01) input data between 0 and
% 2𝜋. As the output, first generate random values 𝜉𝑖, distributed uniformly
% between [0, 0.1], with the same size of the input vector and then calculate
% the to-be-regressed output via the equation
% 𝑦𝑖 = sin(𝑥𝑖) + 𝜉𝑖
x_i = 0:0.01:2*pi;
xi_i = rand(1, length(x_i)) * 0.1;
y_i = sin(x_i) + xi_i;

%%
% Split the dataset into a train and test sets, 𝑥train, 𝑦train, 𝑥test and 𝑦test, using
% a 80/20 % split. To do this, you can use the function
% cvpartition().

cv = cvpartition(length(x_i), 'HoldOut', 0.2);

x_train = x_i(training(cv));
y_train = y_i(training(cv));
x_test = x_i(test(cv));
y_test = y_i(test(cv));

%%
% 3. Now we define the neural network parameters; construct a neural network
% with one output and one input layer, with 25 neurons, for the learning
% rate make use of 𝛼 = 0.03, and lastly train the neurons for 1000 epochs
% (iterations).

neurons = 25;
alpha = 0.03;
epochs = 1000;

nn = feedforwardnet(neurons);
nn.trainParam.lr = alpha;
nn.trainParam.epochs = epochs;

nn = train(nn, x_train, y_train);

%%
% 4. During each epoch, by using each input-output pair, 
% calculate the prediction error of the network by using 
% the current neuron weights (forward run) and update 
% the weights based on the prediction error (back-propagation).
% Store the training RMS error of each epoch.

trainingError = zeros(1, epochs);

% Training loop
for epoch = 1:epochs
    % Train the neural network for one epoch
    [nn,tr] = train(nn, x_train, y_train);
    
    % Perform forward run to get predictions for training set
    predictions = nn(x_train);
    
    % Calculate the prediction error for the training set
    error = y_train - predictions;
    
    % Calculate the RMS error for the current epoch
    trainingError(epoch) = sqrt(mean(error.^2));

end

% training_err = zeros(1, epochs);
% 
% for epoch_loop = 1:epochs
%     SSE = 0;
% 
%     for i = 1:length(x_train)
%         output = nn(x_train(i));        
%         error = y_train(i) - output;
%         SSE = SSE + error^2;
%         nn = adapt(nn, x_train(i), y_train(i)); % Back-propagation
%     end
%     
%     % Calculate the RMS error for the current epoch
%     training_err(epoch_loop) = sqrt(SSE / length(x_train));
% end

%% 
% 6. Visualize the true data and the predictions via the obtained NNs for the
% training and test datasets. Create histogram plots of the test errors (of
% different NNs) using the function
% plotterrhist().
% True data and predictions for training dataset
train_predictions = nn(x_train);
train_errors = y_train - train_predictions;
test_predictions = nn(x_test);
test_errors = y_test - test_predictions;

%  true data vs predictions for training dataset
figure(1);
plot(x_train, y_train, 'b', 'LineWidth', 2);
hold on;
plot(x_train, train_predictions, 'r--', 'LineWidth', 2);
hold off;
title('True Data vs. Predictions (Training Dataset)');
legend('True Data', 'Predictions');
xlabel('Input');
ylabel('Output');

% true data vs predictions for test dataset
figure(2);
plot(x_test, y_test, 'b', 'LineWidth', 2);
hold on;
plot(x_test, test_predictions, 'r--', 'LineWidth', 2);
hold off;
title('True Data vs. Predictions (Test Dataset)');
legend('True Data', 'Predictions');
xlabel('Input');
ylabel('Output');

% histogram of test errors
figure(3);
ploterrhist(test_errors);
title('Histogram of Test Errors');
xlabel('Error');
ylabel('Frequency');

%% 
% Another important metric to compare the NN models is the out-of-sample
% prediction error, the prediction error of the model for the input data that
% is considerably ‘outside’ of the training samples. Generate new input
% data between [3𝜋, 5𝜋] with steps of 0.01 and calculate the ‘true’ output
% by 𝑦𝑖 = sin(𝑥𝑖) (there is no need to corrupt this data with random values.).
% Now compare the models obtained from the steps above with respect to
% their out-of-sample prediction errors. Can you regularize the cost function
% such that the sinusoidal behaviour is embedded into the NN?

x_new = 3*pi:0.01:5*pi;
y_true = sin(x_new);

predictions_model_new = nn(x_new);
error_model_new = y_true - predictions_model_new;
rms_error_model_new = sqrt(mean(error_model_new.^2));

% Regarding regularization, one can embed the sinusoidal behavior into 
% the neural network by adjusting the cost function. 
% One common regularization technique is to use weight decay regularization, 
% also known as L2 regularization. 
% This technique adds a penalty term to the cost function that encourages 
% smaller weights in the network, effectively preventing overfitting 
% and promoting smoother functions. 
% Enable weight decay regularization by setting 
% the net.trainParam.weightDecay parameter to a non-zero value 
% before training the network.

nn.trainParam.weightDecay = 0.001;  % Adjust the weight decay parameter as needed
nn_reg = train(nn, x_train, y_train);

prediction_regularized = nn_reg(x_train);