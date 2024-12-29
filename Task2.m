% One-vs-Rest Logistic Regression with Second-Order Polynomial Decision Boundaries

% Load the data
train_features = readmatrix('Train_Validation_InputFeatures.xlsx');
train_target = readtable('Train_Validation_TargetValue.xlsx');
test_features = readmatrix('Test_InputFeatures.xlsx');
test_target = readtable('Test_TargetValue.xlsx');

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

% Add polynomial features (second-order)
train_features_poly = addPolynomialFeatures(train_features);
test_features_poly = addPolynomialFeatures(test_features);

% Logistic regression parameters
[m, n] = size(train_features_poly);
theta = zeros(n + 1, num_classes); % Initialize weights (+1 for intercept)

% Add intercept term to features
train_features_poly = [ones(m, 1), train_features_poly];
test_features_poly = [ones(size(test_features_poly, 1), 1), test_features_poly];

% Train one logistic regression model per class
for c = 1:num_classes
    % Define options for optimization
    options = optimset('GradObj', 'on', 'MaxIter', 400);

    % Initialize initial weights
    initial_theta = zeros(n + 1, 1);

    % Optimize using fminunc
    [theta(:, c), ~] = fminunc(@(t)(costFunction(t, train_features_poly, train_labels(:, c))), initial_theta, options);
end

% Predict on test set
predictions = sigmoid(test_features_poly * theta);
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
