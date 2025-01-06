clc;
clear;

% Optimized Softmax Regression with SGD for Second-Order Polynomial Decision Boundaries

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

% Add polynomial features (second-order)
train_features_poly = addPolynomialFeatures(train_features);
test_features_poly = addPolynomialFeatures(test_features);

% Logistic regression parameters
[m, n] = size(train_features_poly);
Theta = zeros(n + 1, num_classes); % Initialize weights

% Add intercept term to features
train_features_poly = [ones(m, 1), train_features_poly];
test_features_poly = [ones(size(test_features_poly, 1), 1), test_features_poly];

% Stochastic Gradient Descent (SGD) parameters
alpha = 0.01; % Learning rate
num_epochs = 50; % Number of passes through the data
batch_size = 256; % Mini-batch size

% Training with SGD
for epoch = 1:num_epochs
    % Shuffle data
    idx = randperm(m);
    train_features_poly = train_features_poly(idx, :);
    train_labels = train_labels(idx, :);

    for i = 1:batch_size:m
        % Mini-batch
        batch_end = min(i + batch_size - 1, m);
        X_batch = train_features_poly(i:batch_end, :);
        Y_batch = train_labels(i:batch_end, :);

        % Compute logits and probabilities
        logits = X_batch * Theta;
        probabilities = softmax(logits);

        % Compute gradient
        gradient = -(1 / size(X_batch, 1)) * (X_batch' * (Y_batch - probabilities));

        % Update weights
        Theta = Theta - alpha * gradient;
    end
end

% Predict on test set
logits = test_features_poly * Theta;
predictions = softmax(logits);
[~, predicted_class] = max(predictions, [], 2);

% Convert numerical predictions to class labels
predicted_labels = classes(predicted_class);

% Calculate accuracy
correct_predictions = strcmp(predicted_labels, test_target.Status);
accuracy = mean(correct_predictions) * 100;

fprintf('Test Accuracy: %.2f%%\n', accuracy);

% Calculate confusion matrix
conf_matrix = confusionmat(test_target.Status, predicted_labels);
disp('Confusion Matrix:');
disp(conf_matrix);

% Calculate precision, recall, and F1 score
precision = zeros(1, num_classes);
recall = zeros(1, num_classes);
f1_score = zeros(1, num_classes);

for i = 1:num_classes
    tp = conf_matrix(i, i);
    fp = sum(conf_matrix(:, i)) - tp;
    fn = sum(conf_matrix(i, :)) - tp;

    precision(i) = tp / (tp + fp);
    recall(i) = tp / (tp + fn);
    f1_score(i) = 2 * (precision(i) * recall(i)) / (precision(i) + recall(i));
end

% Display precision, recall, and F1 score for each class
for i = 1:num_classes
    fprintf('Class %s:\n', classes{i});
    fprintf('  Precision: %.2f\n', precision(i) * 100);
    fprintf('  Recall: %.2f\n', recall(i) * 100);
    fprintf('  F1 Score: %.2f\n', f1_score(i) * 100);
end

% Display macro-averaged metrics
macro_precision = mean(precision);
macro_recall = mean(recall);
macro_f1_score = mean(f1_score);

fprintf('Macro-averaged Metrics:\n');
fprintf('  Precision: %.2f\n', macro_precision * 100);
fprintf('  Recall: %.2f\n', macro_recall * 100);
fprintf('  F1 Score: %.2f\n', macro_f1_score * 100);

% Function to add second-order polynomial features
function polyFeatures = addPolynomialFeatures(X)
    [m, n] = size(X);
    polyFeatures = X;
    
    % Add pairwise and squared terms
    for i = 1:n
        for j = i:n
            polyFeatures = [polyFeatures, X(:, i) .* X(:, j)];
        end
    end
end

% Softmax function
function probabilities = softmax(logits)
    exp_logits = exp(logits - max(logits, [], 2)); % Numerical stability adjustment
    probabilities = exp_logits ./ sum(exp_logits, 2);
end
