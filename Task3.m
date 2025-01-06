clc;
clear;

% Softmax Regression with Linear Decision Boundaries

% Load the data
train_features = readmatrix('Train_Validation_InputFeatures.xlsx');
train_target = readtable('Train_Validation_TargetValue.xlsx');
test_features = readmatrix('Test_InputFeatures.xlsx');
test_target = readtable('Test_TargetValue.xlsx');

% Extract unique classes from the target labels
classes = unique(train_target.Status);
num_classes = length(classes);

% Convert categorical labels to one-hot encoding
train_labels = zeros(size(train_target.Status, 1), num_classes);
test_labels = zeros(size(test_target.Status, 1), num_classes);

for i = 1:num_classes
    train_labels(:, i) = strcmp(train_target.Status, classes{i});
    test_labels(:, i) = strcmp(test_target.Status, classes{i});
end

% Logistic regression parameters
[m, n] = size(train_features);
initial_theta = zeros((n + 1) * num_classes, 1); % Flattened weights

% Add intercept term to features
train_features = [ones(m, 1), train_features];
test_features = [ones(size(test_features, 1), 1), test_features];

% Optimize using fminunc
options = optimset('GradObj', 'on', 'MaxIter', 400);
[theta, ~] = fminunc(@(t)(softmaxCostFunction(t, train_features, train_labels, num_classes)), initial_theta, options);

% Reshape theta back to original dimensions
Theta = reshape(theta, n + 1, num_classes);

% Predict on test set
logits = test_features * Theta;
predictions = softmax(logits);
[~, predicted_class] = max(predictions, [], 2);

% Convert numerical predictions to class labels
predicted_labels = classes(predicted_class);

% Calculate accuracy
correct_predictions = strcmp(predicted_labels, test_target.Status);
accuracy = mean(correct_predictions) * 100;

% Calculate confusion matrix
conf_matrix = confusionmat(test_target.Status, predicted_labels);

% Calculate precision, recall, and F1 score for each class
precision = diag(conf_matrix) ./ sum(conf_matrix, 1)';
recall = diag(conf_matrix) ./ sum(conf_matrix, 2);
f1_score = 2 * (precision .* recall) ./ (precision + recall);

% Handle NaN values due to division by zero
precision(isnan(precision)) = 0;
recall(isnan(recall)) = 0;
f1_score(isnan(f1_score)) = 0;

% Macro-Averaging (Average of individual class metrics)
macro_precision = mean(precision);
macro_recall = mean(recall);
macro_f1_score = mean(f1_score);

% Micro-Averaging (Global precision, recall, and F1 score)
total_tp = sum(diag(conf_matrix)); % Sum of true positives
total_fp = sum(sum(conf_matrix, 1)) - total_tp; % False positives
total_fn = sum(sum(conf_matrix, 2)) - total_tp; % False negatives

micro_precision = total_tp / (total_tp + total_fp);
micro_recall = total_tp / (total_tp + total_fn);
micro_f1_score = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall);

% Combine metrics into a single matrix
general_metrics = [macro_precision, macro_recall, macro_f1_score; 
                   micro_precision, micro_recall, micro_f1_score];

% Display results
fprintf('Test Accuracy: %.2f%%\n', accuracy);
fprintf('Confusion Matrix:\n');
disp(conf_matrix);

fprintf('General Metrics (Precision, Recall, F1 Score):\n');
disp(array2table(general_metrics, ...
    'VariableNames', {'Precision', 'Recall', 'F1_Score'}, ...
    'RowNames', {'Macro_Avg', 'Micro_Avg'}));

% Softmax cost function
function [J, grad] = softmaxCostFunction(theta, X, Y, num_classes)
    [m, n] = size(X);
    theta = reshape(theta, n, num_classes);

    % Compute logits and softmax probabilities
    logits = X * theta;
    probabilities = softmax(logits);

    % Compute cost
    J = -(1/m) * sum(sum(Y .* log(probabilities)));

    % Compute gradient
    grad = -(1/m) * (X' * (Y - probabilities));
    grad = grad(:); % Flatten gradient for fminunc
end

% Softmax function
function probabilities = softmax(logits)
    exp_logits = exp(logits - max(logits, [], 2)); % Numerical stability adjustment
    probabilities = exp_logits ./ sum(exp_logits, 2);
end
