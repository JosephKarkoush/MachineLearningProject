clc;
clear;

%% Load Data
% Load training and test data from the provided Excel files
trainFeatures = readmatrix('Train_Validation_InputFeatures.xlsx');
trainLabels = readtable('Train_Validation_TargetValue.xlsx');
testFeatures = readmatrix('Test_InputFeatures.xlsx');
testLabels = readtable('Test_TargetValue.xlsx');

% Convert labels to categorical for classification
trainLabels = categorical(trainLabels.Status);
testLabels = categorical(testLabels.Status);

%% Genetic Algorithm Setup
% Define the objective function for GA optimization
objectiveFcn = @(params) knnCrossValObjective(params, trainFeatures, trainLabels);

% Define parameter bounds
lb = [1, 1]; % Minimum k = 1, distance type index = 1
ub = [200, 3]; % Maximum k = 200, distance type index = 3

% Genetic Algorithm options
options = optimoptions('ga', ...
    'MaxGenerations', 50, ...
    'PopulationSize', 30, ...
    'Display', 'iter');

% Run Genetic Algorithm
[bestParams, bestAccuracy] = ga(objectiveFcn, 2, [], [], [], [], lb, ub, [], options);

% Extract best parameters
bestK = round(bestParams(1));
distanceTypes = {'cityblock', 'chebychev', 'euclidean'};
bestDistanceType = distanceTypes{round(bestParams(2))};

fprintf('Best k: %d\n', bestK);
fprintf('Best distance type: %s\n', bestDistanceType);

%% Train Final KNN Model
finalModel = fitcknn(trainFeatures, trainLabels, ...
    'NumNeighbors', bestK, ...
    'Distance', bestDistanceType);

%% Test the Model
predictedLabels = predict(finalModel, testFeatures);
accuracy = sum(predictedLabels == testLabels) / numel(testLabels);

fprintf('Test Accuracy: %.2f%%\n', accuracy * 100);

%% Objective Function for Cross-Validation
function accuracy = knnCrossValObjective(params, features, labels)
    k = round(params(1));
    distanceTypes = {'cityblock', 'chebychev', 'euclidean'};
    distance = distanceTypes{round(params(2))};

    % Perform k-fold cross-validation
    cv = cvpartition(labels, 'KFold', 5);
    accuracies = zeros(cv.NumTestSets, 1);

    for i = 1:cv.NumTestSets
        trainIdx = training(cv, i);
        testIdx = test(cv, i);

        % Train KNN model
        model = fitcknn(features(trainIdx, :), labels(trainIdx), ...
            'NumNeighbors', k, 'Distance', distance);

        % Evaluate on validation set
        predictions = predict(model, features(testIdx, :));
        accuracies(i) = sum(predictions == labels(testIdx)) / numel(predictions);
    end

    % Return the negative mean accuracy as the objective to minimize
    accuracy = -mean(accuracies);
end
