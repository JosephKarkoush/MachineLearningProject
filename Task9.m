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
fitnessFunction = @(x) svmFitnessMultiClass(x, train_features, train_labels_num);

% Set genetic algorithm options
options = optimoptions('ga', ...
    'PopulationSize', 20, ...
    'MaxGenerations', 50, ...
    'Display', 'iter', ...
    'UseParallel', true);

% Define bounds for hyperparameters (C and σ)
lb = [0.0001, 0.0001];
ub = [1000, 1000];

% Run genetic algorithm to optimize C and σ
[optimalParams, ~] = ga(fitnessFunction, 2, [], [], [], [], lb, ub, [], options);
optimalC = optimalParams(1);
optimalSigma = optimalParams(2);

fprintf('Optimal Parameters:\nC = %.4f\nSigma = %.4f\n', optimalC, optimalSigma);

% Train the final SVM model using optimal parameters
finalModel = fitcecoc(train_features, train_labels_num, ...
    'Learners', templateSVM('KernelFunction', 'rbf', ...
                            'BoxConstraint', optimalC, ...
                            'KernelScale', optimalSigma));

% Test the final model
predicted_labels = predict(finalModel, test_features);

% Evaluate performance
confMat = confusionmat(test_labels_num, predicted_labels);

% Calculate metrics
accuracy = sum(diag(confMat)) / sum(confMat(:));
precision = diag(confMat) ./ sum(confMat, 1)';
recall = diag(confMat) ./ sum(confMat, 2);
f1Score = 2 * (precision .* recall) ./ (precision + recall);

% Display confusion matrix
disp('Confusion Matrix:');
disp(confMat);

% Display performance metrics
fprintf('Performance Metrics:\n');
fprintf('Accuracy: %.2f%%\n', accuracy * 100);
fprintf('Precision: %.2f\n', mean(precision, 'omitnan'));
fprintf('Recall: %.2f\n', mean(recall, 'omitnan'));
fprintf('F1 Score: %.2f\n', mean(f1Score, 'omitnan'));

% Helper function for SVM fitness evaluation (multi-class)
function fitness = svmFitnessMultiClass(params, features, labels)
    C = params(1);
    sigma = params(2);
    % 5-Fold Cross Validation
    cv = cvpartition(labels, 'KFold', 5);
    accuracy = zeros(cv.NumTestSets, 1);
    for i = 1:cv.NumTestSets
        trainIdx = training(cv, i);
        testIdx = test(cv, i);
        model = fitcecoc(features(trainIdx, :), labels(trainIdx), ...
            'Learners', templateSVM('KernelFunction', 'rbf', ...
                                    'BoxConstraint', C, ...
                                    'KernelScale', sigma));
        predictions = predict(model, features(testIdx, :));
        accuracy(i) = sum(predictions == labels(testIdx)) / length(predictions);
    end
    fitness = -mean(accuracy); % Negative because GA minimizes the function
end
