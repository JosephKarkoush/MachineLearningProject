clc;
clear;

%% Load Data
% Load training and testing data
trainInput = readmatrix('Train_Validation_InputFeatures.xlsx');
trainTarget = readtable('Train_Validation_TargetValue.xlsx');
testInput = readmatrix('Test_InputFeatures.xlsx');
testTarget = readtable('Test_TargetValue.xlsx');

% Convert categorical target values to numeric labels
categories = unique(trainTarget.Status);
numClasses = numel(categories);
trainLabels = zeros(size(trainTarget.Status, 1), 1);
testLabels = zeros(size(testTarget.Status, 1), 1);

for i = 1:numClasses
    trainLabels(strcmp(trainTarget.Status, categories{i})) = i;
    testLabels(strcmp(testTarget.Status, categories{i})) = i;
end

%% Polynomial Feature Expansion
function polyFeatures = expandPolynomialFeatures(X)
    % Add polynomial features (up to second order)
    n = size(X, 2);
    polyFeatures = X;
    for i = 1:n
        for j = i:n
            polyFeatures = [polyFeatures, X(:, i) .* X(:, j)];
        end
    end
end

trainInputPoly = expandPolynomialFeatures(trainInput);
testInputPoly = expandPolynomialFeatures(testInput);

%% Logistic Regression Training (One-vs-Rest)
models = cell(numClasses, 1);
options = optimoptions('fminunc', 'GradObj', 'on', 'MaxIter', 400);

for i = 1:numClasses
    fprintf('Training classifier for class %s...\n', categories{i});
    initialTheta = zeros(size(trainInputPoly, 2), 1);
    binaryLabels = (trainLabels == i);

    % Compute class weights
    weight = sum(binaryLabels == 0) / sum(binaryLabels == 1);
    classWeights = binaryLabels * weight + (1 - binaryLabels);

    [theta, ~] = fminunc(@(t)(costFunctionWeighted(t, trainInputPoly, binaryLabels, classWeights)), initialTheta, options);
    models{i} = theta;
end

%% Prediction Function
function pred = predictOneVsRest(models, X)
    numClasses = numel(models);
    probabilities = zeros(size(X, 1), numClasses);
    for i = 1:numClasses
        probabilities(:, i) = sigmoid(X * models{i});
    end
    [~, pred] = max(probabilities, [], 2);
end

%% Evaluate Performance
predTest = predictOneVsRest(models, testInputPoly);
confusionMat = confusionmat(testLabels, predTest);
accuracy = sum(diag(confusionMat)) / sum(confusionMat(:));

precision = diag(confusionMat) ./ sum(confusionMat, 1)';
recall = diag(confusionMat) ./ sum(confusionMat, 2);
f1Score = 2 * (precision .* recall) ./ (precision + recall);

% Calculate overall metrics
overallPrecision = mean(precision, 'omitnan');
overallRecall = mean(recall, 'omitnan');
overallF1Score = mean(f1Score, 'omitnan');

%% Display Results
fprintf('Confusion Matrix:\n');
confusionMat
fprintf('Accuracy: %.2f%%\n', accuracy * 100);
fprintf('Overall Precision: %.2f\n', overallPrecision);
fprintf('Overall Recall: %.2f\n', overallRecall);
fprintf('Overall F1 Score: %.2f\n', overallF1Score);

%% Helper Functions
function [J, grad] = costFunctionWeighted(theta, X, y, weights)
    m = length(y);
    h = sigmoid(X * theta);
    reg_lambda = 1; % Regularization parameter
    J = (1 / m) * sum(weights .* (-y .* log(h) - (1 - y) .* log(1 - h))) + (reg_lambda / (2 * m)) * sum(theta(2:end).^2);
    grad = (1 / m) * (X' * (weights .* (h - y))) + (reg_lambda / m) * [0; theta(2:end)];
end

function g = sigmoid(z)
    g = 1 ./ (1 + exp(-z));
end
