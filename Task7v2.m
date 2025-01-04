% Load data
data_train = table2array(readtable('Train_Validation_InputFeatures.xlsx'));
labels_train = dummyvar(categorical(readtable('Train_Validation_TargetValue.xlsx').Status));
data_test = table2array(readtable('Test_InputFeatures.xlsx'));
labels_test = categorical(readtable('Test_TargetValue.xlsx').Status);

% Normalize input data
[data_train, mu, sigma] = zscore(data_train);
data_test = (data_test - mu) ./ sigma;

% K-fold cross-validation setup
k = 5;
kfold = cvpartition(size(labels_train, 1), 'KFold', k);

% Hyperparameter bounds
lb = [1, 50, 1, 0.0001]; % Reduce upper bounds
ub = [3, 200, 3, 0.1];

% Genetic Algorithm settings
opts = optimoptions('ga', 'MaxGenerations', 20, 'PopulationSize', 10, ...
    'Display', 'iter', 'UseParallel', true);

% Fitness function
fitnessFcn = @(x) crossValidationFitness(x, data_train, labels_train, kfold);
[bestParams, ~] = ga(fitnessFcn, 4, [], [], [], [], lb, ub, [], opts);

% Extract optimal hyperparameters
numHiddenLayers = round(bestParams(1));
numNeurons = round(bestParams(2));
activationFuncIndex = round(bestParams(3));
regularizationLambda = bestParams(4);
activationFunctions = {'logsig', 'poslin', 'tansig'};
activationFunc = activationFunctions{activationFuncIndex};

% Train final neural network
net = patternnet(repmat(numNeurons, 1, numHiddenLayers), 'trainrp');
for i = 1:numHiddenLayers
    net.layers{i}.transferFcn = activationFunc;
end
net.performParam.regularization = regularizationLambda;
[net, tr] = train(net, data_train', labels_train');

% Evaluate on test set
predictions = net(data_test');
[~, predLabels] = max(predictions, [], 1);

% Convert categorical test labels to numeric
actualLabels = double(labels_test);

% Calculate confusion matrix
confMatrix = confusionmat(actualLabels, predLabels');

% Calculate per-class metrics
precision = diag(confMatrix) ./ sum(confMatrix, 2);
recall = diag(confMatrix) ./ sum(confMatrix, 1)';
f1score = 2 * (precision .* recall) ./ (precision + recall);

% Handle NaN cases
precision(isnan(precision)) = 0;
recall(isnan(recall)) = 0;
f1score(isnan(f1score)) = 0;

% Calculate overall metrics
macroPrecision = mean(precision);
macroRecall = mean(recall);
macroF1Score = mean(f1score);

% Micro-average metrics
totalTruePositives = sum(diag(confMatrix));
totalPredictedPositives = sum(confMatrix, 'all');
microPrecision = totalTruePositives / totalPredictedPositives;

microRecall = totalTruePositives / sum(sum(confMatrix, 2)); % Same as micro precision here
microF1Score = 2 * (microPrecision * microRecall) / (microPrecision + microRecall);

% Print results
fprintf('Accuracy: %.2f%%\n', mean(predLabels' == actualLabels) * 100);
fprintf('Confusion Matrix:\n');
disp(confMatrix);
fprintf('Macro Precision: %.4f\n', macroPrecision);
fprintf('Macro Recall: %.4f\n', macroRecall);
fprintf('Macro F1 Score: %.4f\n', macroF1Score);
fprintf('Micro Precision: %.4f\n', microPrecision);
fprintf('Micro Recall: %.4f\n', microRecall);
fprintf('Micro F1 Score: %.4f\n', microF1Score);

% Display optimal hyperparameters
fprintf('Optimal Hyperparameters:\n');
fprintf('Number of Hidden Layers: %d\n', numHiddenLayers);
fprintf('Number of Neurons per Layer: %d\n', numNeurons);
fprintf('Activation Function: %s\n', activationFunc);
fprintf('Regularization Lambda: %.5f\n', regularizationLambda);

function fitness = crossValidationFitness(params, data, labels, kfold)
    numHiddenLayers = round(params(1));
    numNeurons = round(params(2));
    activationFuncIndex = round(params(3));
    regularizationLambda = params(4);

    activationFunctions = {'logsig', 'poslin', 'tansig'};
    activationFunc = activationFunctions{activationFuncIndex};

    foldAccuracies = zeros(kfold.NumTestSets, 1);

    parfor fold = 1:kfold.NumTestSets
        trainIdx = training(kfold, fold);
        testIdx = test(kfold, fold);

        net = patternnet(repmat(numNeurons, 1, numHiddenLayers), 'trainrp');
        for i = 1:numHiddenLayers
            net.layers{i}.transferFcn = activationFunc;
        end
        net.performParam.regularization = regularizationLambda;

        net = train(net, data(trainIdx, :)', labels(trainIdx, :)');
        predictions = net(data(testIdx, :)');
        [~, predLabels] = max(predictions, [], 1);
        actualLabels = vec2ind(labels(testIdx, :)');
        foldAccuracies(fold) = mean(predLabels == actualLabels);
    end

    fitness = -mean(foldAccuracies);
end
