clc;
clear;

% One-vs-Rest Logistic Regression Implementation in MATLAB

% Load the data
train_features = readmatrix('Train_Validation_InputFeatures.xlsx');
train_target = readtable('Train_Validation_TargetValue.xlsx');
test_features = readmatrix('Test_InputFeatures.xlsx');
test_target = readtable('Test_TargetValue.xlsx');

% Normalize features (zero mean, unit variance)
mu = mean(train_features);
sigma = std(train_features);
train_features = (train_features - mu) ./ sigma;
test_features = (test_features - mu) ./ sigma;

% Extract unique classes from the target labels
classes = unique(train_target.Status);
num_classes = length(classes);

% Convert categorical labels to numerical for one-vs-rest
train_labels = zeros(size(train_target.Status, 1), num_classes);
test_labels = zeros(size(test_target.Status, 1), num_classes);

for i = 1:num_classes
    train_labels(:, i) = strcmp(train_target.Status, classes{i});
    test_labels(:, i) = strcmp(test_target.Status, classes{i});
end

% Logistic regression parameters
[m, n] = size(train_features);
theta = zeros(n + 1, num_classes); % Initialize weights (+1 for intercept)

% Add intercept term to features
train_features = [ones(m, 1), train_features];
test_features = [ones(size(test_features, 1), 1), test_features];

% Train one logistic regression model per class
for c = 1:num_classes
    % Define options for optimization with increased MaxIter
    options = optimset('GradObj', 'on', 'MaxIter', 1000); % Increased MaxIter

    % Initialize initial weights
    initial_theta = zeros(n + 1, 1);

    % Optimize using fminunc
    [theta(:, c), ~] = fminunc(@(t)(costFunction(t, train_features, train_labels(:, c))), initial_theta, options);
end

% Predict on test set
predictions = sigmoid(test_features * theta);
[~, predicted_class] = max(predictions, [], 2);

% Convert numerical predictions to class labels
predicted_labels = classes(predicted_class);

% Calculate confusion matrix
true_labels_numeric = grp2idx(test_target.Status); % Convert true labels to numeric
predicted_labels_numeric = grp2idx(predicted_labels); % Convert predicted labels to numeric

conf_matrix = confusionmat(true_labels_numeric, predicted_labels_numeric);

disp('Confusion Matrix:');
disp(conf_matrix);

% Calculate overall metrics
TP = sum(diag(conf_matrix));
FP = sum(sum(conf_matrix, 1)) - TP;
FN = sum(sum(conf_matrix, 2)) - TP;
TN = sum(conf_matrix(:)) - (TP + FP + FN);

% Overall precision, recall, and F1 score
precision = TP / (TP + FP);
recall = TP / (TP + FN);
f1_score = 2 * (precision * recall) / (precision + recall);

% Calculate overall accuracy
accuracy = sum(diag(conf_matrix)) / sum(conf_matrix(:)) * 100;

% Display metrics
fprintf('Overall Metrics:\n');
fprintf('  Accuracy: %.2f%%\n', accuracy);
fprintf('  Precision: %.2f\n', precision);
fprintf('  Recall: %.2f\n', recall);
fprintf('  F1 Score: %.2f\n', f1_score);

% Logistic regression cost function
function [J, grad] = costFunction(theta, X, y)
    m = length(y); % Number of training examples

    % Hypothesis
    h = sigmoid(X * theta);

    % Cost function
    J = (1/m) * (-y' * log(h) - (1 - y)' * log(1 - h));

    % Gradient
    grad = (1/m) * (X' * (h - y));
end

% Sigmoid function
function g = sigmoid(z)
    g = 1 ./ (1 + exp(-z));
end
