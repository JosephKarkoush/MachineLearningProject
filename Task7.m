% Load data
train_input = readtable('Train_Validation_InputFeatures.xlsx');
train_target = readtable('Train_Validation_TargetValue.xlsx');
test_input = readtable('Test_InputFeatures.xlsx');
test_target = readtable('Test_TargetValue.xlsx');

% Convert tables to arrays
train_input = table2array(train_input);
train_target = table2array(train_target);
test_input = table2array(test_input);
test_target = table2array(test_target);

% Convert categorical target values to one-hot encoding
if iscell(train_target)
    train_target = cellfun(@str2double, train_target); % Convert cell to numeric if necessary
end
if iscell(test_target)
    test_target = cellfun(@str2double, test_target); % Convert cell to numeric if necessary
end

train_target = round(train_target); % Ensure integer values
test_target = round(test_target);

uniqueClasses = unique(train_target);
if ~isequal(uniqueClasses, 1:numel(uniqueClasses))
    [~, ~, train_target] = unique(train_target); % Relabel classes to 1, 2, 3, ...
    [~, ~, test_target] = unique(test_target);
end

numClasses = numel(unique(train_target));
train_target = ind2vec(train_target'); % Convert to one-hot
test_target = ind2vec(test_target');   % Convert to one-hot

% Define the genetic algorithm objective function
function error = objectiveFcn(params, train_input, train_target)
    numLayers = round(params(1));
    numNeurons = round(params(2));
    activationFcnIdx = round(params(3));
    lambda = params(4);
    activationFcns = {'logsig', 'relu', 'tansig'};
    activationFcn = activationFcns{activationFcnIdx};

    % Define neural network structure
    net = patternnet(repmat(numNeurons, 1, numLayers));
    for i = 1:numLayers
        net.layers{i}.transferFcn = activationFcn;
    end

    % Set regularization parameter
    net.performParam.regularization = lambda;

    % Perform k-fold cross-validation
    k = 5;
    cv = cvpartition(size(train_input, 1), 'KFold', k);
    accuracies = zeros(k, 1);

    for fold = 1:k
        trainIdx = training(cv, fold);
        valIdx = test(cv, fold);

        net = train(net, train_input(trainIdx, :)', train_target(:, trainIdx));
        predictions = net(train_input(valIdx, :)');

        % Calculate accuracy
        [~, predLabels] = max(predictions);
        [~, trueLabels] = max(train_target(:, valIdx));
        accuracies(fold) = sum(predLabels == trueLabels) / length(trueLabels);
    end

    % Objective function value: negative mean accuracy
    error = -mean(accuracies);
end

% Set bounds for hyperparameters
lb = [1, 1, 1, 0]; % Min values: 1 layer, 1 neuron, sigmoid activation, 0 regularization
ub = [3, 400, 3, 1]; % Max values: 3 layers, 400 neurons, 3 activations, max reg.

% Perform optimization
options = optimoptions('ga', 'Display', 'iter', 'PopulationSize', 20, 'MaxGenerations', 50);
[optParams, ~] = ga(@(params) objectiveFcn(params, train_input, train_target), ...
                    4, [], [], [], [], lb, ub, [], options);

% Extract optimal hyperparameters
optLayers = round(optParams(1));
optNeurons = round(optParams(2));
optActivationFcnIdx = round(optParams(3));
optLambda = optParams(4);
activationFcns = {'logsig', 'relu', 'tansig'};
optActivationFcn = activationFcns{optActivationFcnIdx};

fprintf('Optimal Hyperparameters:\n');
fprintf('Number of Layers: %d\n', optLayers);
fprintf('Number of Neurons per Layer: %d\n', optNeurons);
fprintf('Activation Function: %s\n', optActivationFcn);
fprintf('Regularization Parameter: %.4f\n', optLambda);

% Train final model
finalNet = patternnet(repmat(optNeurons, 1, optLayers));
for i = 1:optLayers
    finalNet.layers{i}.transferFcn = optActivationFcn;
end
finalNet.performParam.regularization = optLambda;
finalNet = train(finalNet, train_input', train_target);

% Test the final model
predictions = finalNet(test_input');
[~, predLabels] = max(predictions);
[~, trueLabels] = max(test_target);

% Evaluate performance
confMatrix = confusionmat(trueLabels, predLabels);
accuracy = sum(diag(confMatrix)) / sum(confMatrix(:));
precision = diag(confMatrix) ./ sum(confMatrix, 2);
recall = diag(confMatrix) ./ sum(confMatrix, 1)';
F1 = 2 * (precision .* recall) ./ (precision + recall);

fprintf('Confusion Matrix:\n');
disp(confMatrix);
fprintf('Accuracy: %.4f\n', accuracy);
fprintf('Precision: %.4f\n', mean(precision, 'omitnan'));
fprintf('Recall: %.4f\n', mean(recall, 'omitnan'));
fprintf('F1 Score: %.4f\n', mean(F1, 'omitnan'));