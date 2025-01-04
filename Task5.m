% Load the training-validation and test data
train_input = readtable('Train_Validation_InputFeatures.xlsx');
train_labels = readtable('Train_Validation_TargetValue.xlsx');
test_input = readtable('Test_InputFeatures.xlsx');
test_labels = readtable('Test_TargetValue.xlsx');

% Convert tables to matrices
X_train = table2array(train_input);
y_train = categorical(train_labels.Status);
X_test = table2array(test_input);
y_test = categorical(test_labels.Status);

% Normalize the input features
X_train = normalize(X_train);
X_test = normalize(X_test);

% Define the objective function for the genetic algorithm
knn_objective = @(params) knnObjectiveFunction(params, X_train, y_train);

% Set up genetic algorithm options
lb = [1, 1]; % Lower bounds (k, distance type index)
ub = [200, 3]; % Upper bounds (k, distance type index)
options = optimoptions('ga', 'MaxGenerations', 50, 'PopulationSize', 20, 'Display', 'iter');

% Run genetic algorithm
optimal_params = ga(knn_objective, 2, [], [], [], [], lb, ub, [], options);

% Extract optimal hyperparameters
k_optimal = round(optimal_params(1));
% Define the possible distance metrics
distance_metrics = {'cityblock', 'chebychev', 'euclidean'};

% Use proper indexing to select the optimal distance metric
distance_metric_optimal = distance_metrics{round(optimal_params(2))};

% Train final KNN model
final_knn = fitcknn(X_train, y_train, 'NumNeighbors', k_optimal, 'Distance', distance_metric_optimal);

% Evaluate performance on test data
y_pred_test = predict(final_knn, X_test);
conf_matrix = confusionmat(y_test, y_pred_test);

% Calculate evaluation metrics
accuracy = mean(y_pred_test == y_test);
precision = diag(conf_matrix) ./ sum(conf_matrix, 2);
recall = diag(conf_matrix) ./ sum(conf_matrix, 1)';
F1 = 2 * (precision .* recall) ./ (precision + recall);

% Display results
fprintf('Optimal Number of Neighbors: %d\n', k_optimal);
fprintf('Optimal Distance Metric: %s\n', distance_metric_optimal);

fprintf('Confusion Matrix:\n');
disp(conf_matrix);

fprintf('Accuracy: %.2f\n', accuracy);
fprintf('Precision: %.2f\n', mean(precision, 'omitnan'));
fprintf('Recall: %.2f\n', mean(recall, 'omitnan'));
fprintf('F1 Score: %.2f\n', mean(F1, 'omitnan'));

% Define the objective function for cross-validation
function score = knnObjectiveFunction(params, X, y)
    k = round(params(1));
    distance_metrics = {'cityblock', 'chebychev', 'euclidean'};
    distance_type = distance_metrics{round(params(2))};
    
    % Perform k-fold cross-validation
    cv = cvpartition(y, 'KFold', 5);
    acc = zeros(cv.NumTestSets, 1);
    
    for i = 1:cv.NumTestSets
        X_train_cv = X(training(cv, i), :);
        y_train_cv = y(training(cv, i));
        X_valid_cv = X(test(cv, i), :);
        y_valid_cv = y(test(cv, i));
        
        mdl = fitcknn(X_train_cv, y_train_cv, 'NumNeighbors', k, 'Distance', distance_type);
        y_pred = predict(mdl, X_valid_cv);
        acc(i) = mean(y_pred == y_valid_cv);
    end
    
    % Minimize the negative mean accuracy
    score = -mean(acc);
end
