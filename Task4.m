% Optimized Softmax Regression with Second-Order Polynomial Decision Boundaries

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
initial_theta = zeros((n + 1) * num_classes, 1); % Flattened weights

% Add intercept term to features
train_features_poly = [ones(m, 1), train_features_poly];
test_features_poly = [ones(size(test_features_poly, 1), 1), test_features_poly];

% Optimize using a mini-batch approach to prevent memory overflow
batch_size = 1000; % Adjust based on system memory
num_batches = ceil(m / batch_size);

options = optimset('GradObj', 'on', 'MaxIter', 100); % Reduce MaxIter for testing

Theta = zeros(n + 1, num_classes); % Initialize Theta
for batch = 1:num_batches
    % Define batch indices
    batch_start = (batch - 1) * batch_size + 1;
    batch_end = min(batch * batch_size, m);

    % Extract batch data
    X_batch = train_features_poly(batch_start:batch_end, :);
    Y_batch = train_labels(batch_start:batch_end, :);

    % Optimize using fminunc
    [theta, ~] = fminunc(@(t)(softmaxCostFunction(t, X_batch, Y_batch, num_classes)), initial_theta, options);

    % Accumulate results
    Theta = Theta + reshape(theta, n + 1, num_classes);
end

Theta = Theta / num_batches; % Average the results across batches

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
