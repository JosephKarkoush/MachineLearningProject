function optimizeSVM()
    %% Load Data
    % Load Training/Validation and Testing Data
    trainValidationFeatures = readmatrix('Train_Validation_InputFeatures.xlsx');
    trainValidationLabels = readtable('Train_Validation_TargetValue.xlsx');
    testFeatures = readmatrix('Test_InputFeatures.xlsx');
    testLabels = readtable('Test_TargetValue.xlsx');

    % Convert categorical labels to numerical if necessary
    categories = unique(trainValidationLabels.Status);
    trainLabelsNumerical = grp2idx(trainValidationLabels.Status); % Changed name for consistency
    testLabelsNumerical = grp2idx(testLabels.Status);

    %% Apply PCA for Dimensionality Reduction
    [coeff, trainValidationFeaturesPCA, ~, ~, explained] = pca(trainValidationFeatures);
    testFeaturesPCA = testFeatures * coeff;

    % Retain components that explain 95% of variance
    explainedVarianceThreshold = 95;
    numComponents = find(cumsum(explained) >= explainedVarianceThreshold, 1);
    trainValidationFeaturesPCA = trainValidationFeaturesPCA(:, 1:numComponents);
    testFeaturesPCA = testFeaturesPCA(:, 1:numComponents);

    %% Genetic Algorithm Hyperparameter Optimization
    % Define fitness function for GA
    kFold = 5; % Number of folds for cross-validation

    fitnessFunction = @(x) svmFitnessFunction(x, trainValidationFeaturesPCA, trainLabelsNumerical, kFold);

    % Define bounds for [C, sigma]
    lb = [0.01, 0.01];
    ub = [100, 100];

        % Run Genetic Algorithm with Parallelization
    options = optimoptions('ga', 'PopulationSize', 20, 'MaxGenerations', 10, 'Display', 'iter', ...
                           'UseParallel', true);
    [optimizedParams, ~] = ga(fitnessFunction, 2, [], [], [], [], lb, ub, [], options);

    C_opt = optimizedParams(1);
    sigma_opt = optimizedParams(2);

    % Print final optimal hyperparameters
    fprintf('Optimal Box Constraint (C): %.4f\n', C_opt);
    fprintf('Optimal Kernel Scale (σ): %.4f\n', sigma_opt);


    %% Train Final SVM Model
    template = templateSVM('KernelFunction', 'rbf', 'BoxConstraint', C_opt, 'KernelScale', sigma_opt);
    SVMModel = fitcecoc(trainValidationFeaturesPCA, trainLabelsNumerical, 'Learners', template);

    %% Test Model
    predictedLabels = predict(SVMModel, testFeaturesPCA);

    % Compute Metrics
    confusionMat = confusionmat(testLabelsNumerical, predictedLabels);
    accuracy = sum(diag(confusionMat)) / sum(confusionMat(:));
    precision = diag(confusionMat) ./ sum(confusionMat, 2);
    recall = diag(confusionMat) ./ sum(confusionMat, 1)';
    f1Score = 2 * (precision .* recall) ./ (precision + recall);

    %% Display Results
    fprintf('Confusion Matrix:\\n');
    disp(confusionMat);

    fprintf('Accuracy: %.2f%%\\n', accuracy * 100);

    fprintf('Precision by class:\\n');
    disp(precision);

    fprintf('Recall by class:\\n');
    disp(recall);

    fprintf('F1 Score by class:\\n');
    disp(f1Score);
end

%% Fitness Function for GA
function fitness = svmFitnessFunction(params, trainFeatures, trainLabels, kFold)
    % Persistent cache for fitness results
    persistent cache;
    if isempty(cache)
        cache = containers.Map('KeyType', 'char', 'ValueType', 'double');
    end

    % Extract parameters
    C = params(1);
    sigma = params(2);

    % Create a unique key for caching
    key = mat2str(params);
    if isKey(cache, key)
        fitness = cache(key);
        return;
    end

    % Cross-validation
    cv = cvpartition(trainLabels, 'KFold', kFold);
    accuracyList = zeros(kFold, 1);

    parfor fold = 1:kFold
        trainIdx = training(cv, fold);
        testIdx = test(cv, fold);

        template = templateSVM('KernelFunction', 'rbf', 'BoxConstraint', C, 'KernelScale', sigma);
        SVMModel = fitcecoc(trainFeatures(trainIdx, :), trainLabels(trainIdx), 'Learners', template);

        predictions = predict(SVMModel, trainFeatures(testIdx, :));
        accuracyList(fold) = mean(predictions == trainLabels(testIdx));
    end

    % Compute fitness and cache it
    fitness = -mean(accuracyList); % Return negative mean accuracy (since GA minimizes)
    cache(key) = fitness;
end
