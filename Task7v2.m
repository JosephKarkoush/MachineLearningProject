% MATLAB Script for Neural Network with Genetic Algorithm and Evaluation

% Load training and testing data
data_train = readtable('Train_Validation_InputFeatures.xlsx');
labels_train = readtable('Train_Validation_TargetValue.xlsx');
data_test = readtable('Test_InputFeatures.xlsx');
labels_test = readtable('Test_TargetValue.xlsx');

% Convert tables to numeric matrices
data_train = table2array(data_train);
data_test = table2array(data_test);

% Convert labels to categorical and then to dummy variables
labels_train = categorical(labels_train.Status);
labels_test = categorical(labels_test.Status);
labels_train_dummy = dummyvar(labels_train);
labels_test_dummy = dummyvar(labels_test);

% Define k-fold cross-validation
k = 5;
kfold = cvpartition(labels_train, 'KFold', k);

% Define search space for hyperparameters
lb = [1, 1, 1, 0.0001]; % Lower bounds: [#layers, #neurons, activationFunc, lambda]
ub = [3, 400, 3, 1];   % Upper bounds: [#layers, #neurons, activationFunc, lambda]

% Genetic Algorithm Optimization
opts = optimoptions('ga', 'MaxGenerations', 30, 'PopulationSize', 20, ...
    'Display', 'iter', 'UseParallel', true);
fitnessFcn = @(x) crossValidationFitness(x, data_train, labels_train_dummy, kfold);
[bestParams, ~] = ga(fitnessFcn, 4, [], [], [], [], lb, ub, [], opts);

% Extract best hyperparameters
numHiddenLayers = round(bestParams(1));
numNeurons = round(bestParams(2));
activationFuncIndex = round(bestParams(3));
regularizationLambda = bestParams(4);

activationFunctions = {'logsig', 'poslin', 'tansig'};
activationFunc = activationFunctions{activationFuncIndex};

% Print optimal hyperparameters
fprintf('Optimal Hyperparameters:\n');
fprintf('Number of Hidden Layers: %d\n', numHiddenLayers);
fprintf('Number of Neurons per Layer: %d\n', numNeurons);
fprintf('Activation Function: %s\n', activationFunc);
fprintf('Regularization Lambda: %.4f\n', regularizationLambda);

% Train final neural network using the best hyperparameters
net = patternnet(repmat(numNeurons, 1, numHiddenLayers), 'trainscg');
for i = 1:numHiddenLayers
    net.layers{i}.transferFcn = activationFunc;
end
net.performParam.regularization = regularizationLambda;

% Train the network
[net, tr] = train(net, data_train', labels_train_dummy');

% Evaluate on the test set
predictions = net(data_test');
[~, predLabels] = max(predictions, [], 1);
predLabels = categorical(predLabels);

% Calculate performance metrics
confMat = confusionmat(labels_test, predLabels);
accuracy = sum(diag(confMat)) / sum(confMat(:));
precision = diag(confMat) ./ sum(confMat, 2);
recall = diag(confMat) ./ sum(confMat, 1)';
f1Score = 2 * (precision .* recall) ./ (precision + recall);

% Display results
fprintf('Confusion Matrix:\n');
disp(confMat);

fprintf('Accuracy: %.2f\n', accuracy);
fprintf('Precision: %.2f\n', mean(precision));
fprintf('Recall: %.2f\n', mean(recall));
fprintf('F1 Score: %.2f\n', mean(f1Score));

function fitness = crossValidationFitness(params, data, labels, kfold)
    % Cross-validation fitness function for GA
    numHiddenLayers = round(params(1));
    numNeurons = round(params(2));
    activationFuncIndex = round(params(3));
    regularizationLambda = params(4);

    activationFunctions = {'logsig', 'poslin', 'tansig'};
    activationFunc = activationFunctions{activationFuncIndex};

    foldAccuracy = zeros(kfold.NumTestSets, 1);

    for fold = 1:kfold.NumTestSets
        trainIdx = training(kfold, fold);
        testIdx = test(kfold, fold);

        % Train network
        net = patternnet(repmat(numNeurons, 1, numHiddenLayers), 'trainscg');
        for i = 1:numHiddenLayers
            net.layers{i}.transferFcn = activationFunc;
        end
        net.performParam.regularization = regularizationLambda;

        net = train(net, data(trainIdx, :)', labels(trainIdx, :)');

        % Test network
        predictions = net(data(testIdx, :)');
        [~, predLabels] = max(predictions, [], 1);
        actualLabels = vec2ind(labels(testIdx, :)'); % Convert dummy variables to indices

        % Calculate accuracy
        foldAccuracy(fold) = sum(predLabels == actualLabels) / length(testIdx);
    end

    % Fitness is the negative average accuracy (minimization problem)
    fitness = -mean(foldAccuracy);
end
