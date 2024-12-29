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

fprintf('Test Accuracy: %.2f%%\n', accuracy);

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
