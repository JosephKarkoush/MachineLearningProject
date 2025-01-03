%% Load the Data
trainFeatures = readtable('Train_Validation_InputFeatures.xlsx');
trainTargets = readtable('Train_Validation_TargetValue.xlsx');
testFeatures = readtable('Test_InputFeatures.xlsx');
testTargets = readtable('Test_TargetValue.xlsx');

X_train = table2array(trainFeatures);
Y_train = categorical(table2array(trainTargets)); % Convert to categorical
X_test = table2array(testFeatures);
Y_test = categorical(table2array(testTargets)); % Convert to categorical

%% Objective Function for Genetic Algorithm
function [accuracy, net] = evaluateNN(hiddenLayers, neurons, activation, lambda, learningRate, X_train, Y_train, X_val, Y_val)
    layers = [featureInputLayer(size(X_train, 2))];
    
    % Add hidden layers
    for i = 1:hiddenLayers
        % Determine activation function
        if activation == 1
            actLayer = reluLayer;
        elseif activation == 2
            actLayer = tanhLayer;
        else
            actLayer = sigmoidLayer;
        end
        layers = [layers, fullyConnectedLayer(neurons, 'WeightsInitializer', 'he'), actLayer];
    end

    % Add output layer
    layers = [layers, fullyConnectedLayer(numel(categories(Y_train))), softmaxLayer, classificationLayer];

    options = trainingOptions('adam', ...
        'MaxEpochs', 50, ...
        'MiniBatchSize', 64, ...
        'InitialLearnRate', learningRate, ...
        'L2Regularization', lambda, ...
        'Verbose', false, ...
        'Plots', 'none');

    net = trainNetwork(X_train, Y_train, layers, options);
    predictions = classify(net, X_val);
    accuracy = mean(predictions == Y_val);
end

%% Genetic Algorithm
% Bounds for hyperparameters
lb = [1, 1, 1, 0.0001, 0.0001]; % lower bounds [hiddenLayers, neurons, activation, lambda, learningRate]
ub = [3, 400, 3, 1, 0.1];     % upper bounds [hiddenLayers, neurons, activation, lambda, learningRate]

options = optimoptions('ga', ...
    'PopulationSize', 20, ...
    'MaxGenerations', 30, ...
    'Display', 'iter', ...
    'UseParallel', true);

[optimalParams, optimalAccuracy] = ga(@(params) -evaluateNN(round(params(1)), round(params(2)), round(params(3)), params(4), params(5), X_train, Y_train, X_train, Y_train), 5, [], [], [], [], lb, ub, [], options);

%% Train Final Model with Optimal Hyperparameters
hiddenLayers = round(optimalParams(1));
neurons = round(optimalParams(2));
activation = round(optimalParams(3));
if activation == 1
    actFunc = 'relu';
elseif activation == 2
    actFunc = 'tanh';
else
    actFunc = 'sigmoid';
end
lambda = optimalParams(4);
learningRate = optimalParams(5);

% Build final neural network
layers = [featureInputLayer(size(X_train, 2))];
for i = 1:hiddenLayers
    if activation == 1
        actLayer = reluLayer;
    elseif activation == 2
        actLayer = tanhLayer;
    else
        actLayer = sigmoidLayer;
    end
    layers = [layers, fullyConnectedLayer(neurons, 'WeightsInitializer', 'he'), actLayer];
end
layers = [layers, fullyConnectedLayer(numel(categories(Y_train))), softmaxLayer, classificationLayer];

options = trainingOptions('adam', ...
    'MaxEpochs', 50, ...
    'MiniBatchSize', 64, ...
    'InitialLearnRate', learningRate, ...
    'L2Regularization', lambda, ...
    'Verbose', false);

finalNet = trainNetwork(X_train, Y_train, layers, options);
predictions = classify(finalNet, X_test);

%% Evaluate Performance
confusionMatrix = confusionmat(Y_test, predictions);
accuracy = mean(predictions == Y_test);
precision = diag(confusionMatrix) ./ sum(confusionMatrix, 2);
recall = diag(confusionMatrix) ./ sum(confusionMatrix, 1)';
f1Score = 2 * (precision .* recall) ./ (precision + recall);

fprintf('Optimal Hyperparameters:\n');
fprintf('Number of Hidden Layers: %d\n', hiddenLayers);
fprintf('Number of Neurons: %d\n', neurons);
fprintf('Activation Function: %s\n', actFunc);
fprintf('Regularization Parameter: %f\n', lambda);
fprintf('Learning Rate: %f\n', learningRate);
fprintf('\nPerformance Metrics:\n');
fprintf('Accuracy: %f\n', accuracy);
fprintf('Precision: %f\n', mean(precision));
fprintf('Recall: %f\n', mean(recall));
fprintf('F1 Score: %f\n', mean(f1Score));
