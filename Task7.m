% Load Data
train_input = readtable('Train_Validation_InputFeatures.xlsx');
train_target = readtable('Train_Validation_TargetValue.xlsx');
test_input = readtable('Test_InputFeatures.xlsx');
test_target = readtable('Test_TargetValue.xlsx');

% Preprocess Data
X_train = table2array(train_input);
Y_train = grp2idx(train_target.Status); % Convert categorical target to numerical
X_test = table2array(test_input);
Y_test = grp2idx(test_target.Status);

% Normalize Data
X_train = normalize(X_train);
X_test = normalize(X_test);

% Genetic Algorithm Settings
hidden_layers_range = [1, 3];
neurons_range = [1, 400];
activation_functions = {'logsig', 'tansig', 'purelin'}; % Sigmoid, Tanh, ReLU
lambda_range = [1e-5, 1e-1];
learning_rate_range = [1e-4, 1e-1];

% Fitness Function
function fitness = evaluate_hyperparameters(params, X_train, Y_train, activation_functions)
    layers = round(params(1)); % Ensure layers is an integer
    neurons = round(params(2)); % Ensure neurons is an integer
    activation_idx = round(params(3));
    lambda = params(4);
    learning_rate = params(5);

    % Create Neural Network
    net = feedforwardnet(neurons * ones(1, layers)); % Create a network with layers and neurons per layer
    for i = 1:layers
        net.layers{i}.transferFcn = activation_functions{activation_idx};
    end
    net.performParam.regularization = lambda;
    net.trainParam.lr = learning_rate;

    % Train Neural Network
    net.divideParam.trainRatio = 0.7;
    net.divideParam.valRatio = 0.3;
    net.divideParam.testRatio = 0;
    try
        [net, ~] = train(net, X_train', full(ind2vec(Y_train')));
        % Validate Neural Network
        Y_val_pred = net(X_train');
        Y_val_pred = vec2ind(Y_val_pred);
        fitness = -sum(Y_val_pred' == Y_train) / length(Y_train); % Maximize accuracy
    catch
        fitness = Inf; % Handle training errors
    end
end

% Define Genetic Algorithm
ga_options = optimoptions('ga', 'PopulationSize', 50, 'MaxGenerations', 30, 'Display', 'iter');
[param_opt, ~] = ga(@(params)evaluate_hyperparameters(params, X_train, Y_train, activation_functions), 5, [], [], [], [], ...
    [hidden_layers_range(1), neurons_range(1), 1, lambda_range(1), learning_rate_range(1)], ...
    [hidden_layers_range(2), neurons_range(2), length(activation_functions), lambda_range(2), learning_rate_range(2)], ...
    [], ga_options);

% Extract Optimal Parameters
opt_layers = round(param_opt(1));
opt_neurons = round(param_opt(2));
opt_activation = activation_functions{round(param_opt(3))};
opt_lambda = param_opt(4);
opt_learning_rate = param_opt(5);

% Train Final Neural Network
final_net = feedforwardnet(opt_neurons * ones(1, opt_layers));
for i = 1:opt_layers
    final_net.layers{i}.transferFcn = opt_activation;
end
final_net.performParam.regularization = opt_lambda;
final_net.trainParam.lr = opt_learning_rate;
final_net.divideParam.trainRatio = 0.7;
final_net.divideParam.valRatio = 0.3;
final_net.divideParam.testRatio = 0;
[final_net, ~] = train(final_net, X_train', full(ind2vec(Y_train')));

% Test Neural Network
Y_test_pred = final_net(X_test');
Y_test_pred = vec2ind(Y_test_pred);

% Performance Metrics
confusion_mat = confusionmat(Y_test, Y_test_pred);
accuracy = sum(Y_test_pred' == Y_test) / length(Y_test);
precision = diag(confusion_mat) ./ sum(confusion_mat, 2);
recall = diag(confusion_mat) ./ sum(confusion_mat, 1)';
f1_score = 2 * (precision .* recall) ./ (precision + recall);

% Display Results
fprintf('Optimal Hyperparameters:\n');
fprintf('Number of Hidden Layers: %d\n', opt_layers);
fprintf('Number of Neurons per Layer: %d\n', opt_neurons);
fprintf('Activation Function: %s\n', opt_activation);
fprintf('Regularization Parameter: %.5f\n', opt_lambda);
fprintf('Learning Rate: %.5f\n', opt_learning_rate);

fprintf('Performance Metrics:\n');
fprintf('Accuracy: %.2f\n', accuracy);
fprintf('Precision: %.2f\n', mean(precision, 'omitnan'));
fprintf('Recall: %.2f\n', mean(recall, 'omitnan'));
fprintf('F1 Score: %.2f\n', mean(f1_score, 'omitnan'));