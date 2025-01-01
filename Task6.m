% Load the datasets
trainFeatures = readtable('Train_Validation_InputFeatures.xlsx');
trainTarget = readtable('Train_Validation_TargetValue.xlsx');
testFeatures = readtable('Test_InputFeatures.xlsx');
testTarget = readtable('Test_TargetValue.xlsx');

% Convert target variables to categorical if necessary
trainTarget = categorical(trainTarget.Status);
testTarget = categorical(testTarget.Status);

% Define the fitness function for genetic algorithm
function fitness = fitnessFunction(params, trainFeatures, trainTarget)
    maxNumSplits = round(params(1));
    minLeafSize = round(params(2));

    % Perform k-fold cross-validation
    cv = cvpartition(trainTarget, 'KFold', 3); % Reduced to 3 folds
    accuracy = zeros(cv.NumTestSets, 1);

    for i = 1:cv.NumTestSets
        trainIdx = cv.training(i);
        testIdx = cv.test(i);

        % Train a decision tree with the specified parameters
        tree = fitctree(trainFeatures(trainIdx, :), trainTarget(trainIdx), ...
                        'MaxNumSplits', maxNumSplits, 'MinLeafSize', minLeafSize);

        % Evaluate the model
        predictions = predict(tree, trainFeatures(testIdx, :));
        accuracy(i) = mean(predictions == trainTarget(testIdx));
    end

    % Maximize accuracy (minimize negative accuracy for GA)
    fitness = -mean(accuracy);
end

% Set bounds for hyperparameters
lb = [1, 1]; % Lower bounds for maxNumSplits and minLeafSize
ub = [500, 100]; % Reduced upper bounds

% Run genetic algorithm to optimize hyperparameters
opts = optimoptions('ga', 'Display', 'iter', 'UseParallel', true, ...
                    'PopulationSize', 20, 'MaxGenerations', 10); % Reduced population size and generations
if ~isempty(gcp('nocreate'))
    delete(gcp('nocreate')); % Ensure no existing parallel pool
end
parpool('local'); % Enable parallel computing
bestParams = ga(@(params) fitnessFunction(params, trainFeatures, trainTarget), ...
                2, [], [], [], [], lb, ub, [], opts);

% Extract best parameters
bestMaxNumSplits = round(bestParams(1));
bestMinLeafSize = round(bestParams(2));

% Print optimal parameters
fprintf('Optimal MaxNumSplits: %d\n', bestMaxNumSplits);
fprintf('Optimal MinLeafSize: %d\n', bestMinLeafSize);

% Train final model with the best parameters
finalTree = fitctree(trainFeatures, trainTarget, 'MaxNumSplits', bestMaxNumSplits, 'MinLeafSize', bestMinLeafSize);

% Evaluate on the test data
predictions = predict(finalTree, testFeatures);

% Calculate performance metrics
confMat = confusionmat(testTarget, predictions);
accuracy = sum(diag(confMat)) / sum(confMat(:));
precision = diag(confMat) ./ sum(confMat, 2);
recall = diag(confMat) ./ sum(confMat, 1)';
F1 = 2 * (precision .* recall) ./ (precision + recall);

% Display results
fprintf('Confusion Matrix:\n');
disp(confMat);

fprintf('Accuracy: %.2f%%\n', accuracy * 100);
fprintf('Precision: %.2f\n', mean(precision, 'omitnan'));
fprintf('Recall: %.2f\n', mean(recall, 'omitnan'));
fprintf('F1 Score: %.2f\n', mean(F1, 'omitnan'));

% Plot the confusion matrix
figure;
confusionchart(testTarget, predictions);
title('Confusion Matrix');
%%dsdds