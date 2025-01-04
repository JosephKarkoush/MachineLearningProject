% Load datasets
train_features = readmatrix('Train_Validation_InputFeatures.xlsx');
train_labels = readtable('Train_Validation_TargetValue.xlsx');
test_features = readmatrix('Test_InputFeatures.xlsx');
test_labels = readtable('Test_TargetValue.xlsx');

% Convert labels to numerical values if needed
[train_labels_num, label_mapping] = grp2idx(train_labels.Status);
test_labels_num = grp2idx(test_labels.Status);

% Normalize the features
train_features = normalize(train_features);
test_features = normalize(test_features);

% Define fitness function for genetic algorithm
fitnessFunction = @(x) svmFitnessLinear(x, train_features, train_labels_num);

% Set genetic algorithm options
options = optimoptions('ga', ...
    'PopulationSize', 10, ... % Reduced population size
    'MaxGenerations', 20, ... % Reduced number of generations
    'Display', 'iter', ...
    'UseParallel', true); % Ensure parallel computing is enabled

% Define narrower bounds for hyperparameter (C)
lb = 0.01; % Minimum value for C
ub = 100;  % Maximum value for C

% Run genetic algorithm to optimize C
[optimalC, ~] = ga(fitnessFunction, 1, [], [], [], [], lb, ub, [], options);

fprintf('Optimal Box Constraint (C): %.4f\n', optimalC);

% Train the final Linear SVM model using optimal parameters
finalModel = fitcecoc(train_features, train_labels_num, ...
    'Learners', templateSVM('KernelFunction', 'linear', ...
                            'BoxConstraint', optimalC));

% Test the final model
predicted_labels = predict(finalModel, test_features);

% Evaluate performance
confMat = confusionmat(test_labels_num, predicted_labels);
accuracy = sum(diag(confMat)) / sum(confMat(:));
precision = diag(confMat) ./ sum(confMat, 1)';
recall = diag(confMat) ./ sum(confMat, 2);
f1Score = 2 * (precision .* recall) ./ (precision + recall);

fprintf('Performance Metrics:\n');
fprintf('Accuracy: %.2f%%\n', accuracy * 100);
fprintf('Precision: %.2f\n', mean(precision, 'omitnan'));
fprintf('Recall: %.2f\n', mean(recall, 'omitnan'));
fprintf('F1 Score: %.2f\n', mean(f1Score, 'omitnan'));

% Helper function for Linear SVM fitness evaluation
function fitness = svmFitnessLinear(C, features, labels)
    % 3-Fold Cross Validation (faster than 5-fold)
    cv = cvpartition(labels, 'KFold', 3);
    accuracy = zeros(cv.NumTestSets, 1);
    for i = 1:cv.NumTestSets
        trainIdx = training(cv, i);
        testIdx = test(cv, i);
        model = fitcecoc(features(trainIdx, :), label s(trainIdx), ...
            'Learners', templateSVM('KernelFunction', 'linear', ...
                                    'BoxConstraint', C));
        predictions = predict(model, features(testIdx, :));
        accuracy(i) = sum(predictions == labels(testIdx)) / length(predictions);
    end
    fitness = -mean(accuracy); % Negative because GA minimizes the function
end
