clc;
clear all;
% Läs in data från Excel-filerna
train_features = readmatrix('Train_Validation_InputFeatures.xlsx');
train_labels = readtable('Train_Validation_TargetValue.xlsx');
test_features = readmatrix('Test_InputFeatures.xlsx');
test_labels = readtable('Test_TargetValue.xlsx');

% Konvertera tränings- och testetiketter till numeriska värden
class_labels = unique(train_labels.Status);
num_classes = numel(class_labels);
train_labels_numeric = zeros(size(train_labels.Status));
test_labels_numeric = zeros(size(test_labels.Status));

for i = 1:num_classes
    train_labels_numeric(strcmp(train_labels.Status, class_labels{i})) = i;
    test_labels_numeric(strcmp(test_labels.Status, class_labels{i})) = i;
end

% Standardisera funktionerna
mu = mean(train_features);
sigma = std(train_features);
train_features = (train_features - mu) ./ sigma;
test_features = (test_features - mu) ./ sigma;

% Softmax Regression med linjära beslutsgränser
% Anpassar viktmatrisen W och bias vektor b
W = zeros(size(train_features, 2), num_classes); % Viktmatris
b = zeros(1, num_classes); % Biasvektor

% Optimeringsproblem: Anpassa W och b med iterativ lösning
options = optimoptions('fminunc', 'Algorithm', 'trust-region', 'SpecifyObjectiveGradient', true, 'Display', 'off');
softmaxObjective = @(theta) softmaxCostFunction(theta, train_features, train_labels_numeric, num_classes);

% Initiera parametrar
initial_params = [W(:); b(:)];
[opt_params, ~] = fminunc(softmaxObjective, initial_params, options);

% Extrahera optimala vikter och bias
W = reshape(opt_params(1:end-num_classes), size(train_features, 2), num_classes);
b = opt_params(end-num_classes+1:end);

% Förutsägelser på testdata
scores = test_features * W + b; % Linjär kombination
probs = softmax(scores); % Softmax-sannolikheter
[~, predicted_classes] = max(probs, [], 2); % Klass med högst sannolikhet

% Utvärdera prestanda
confusion_matrix = confusionmat(test_labels_numeric, predicted_classes);
accuracy = sum(diag(confusion_matrix)) / sum(confusion_matrix(:));
fprintf('Noggrannhet: %.2f%%\n', accuracy * 100);

% Precision, Recall och F1 Score för varje klass
precision = zeros(num_classes, 1);
recall = zeros(num_classes, 1);
f1_score = zeros(num_classes, 1);

for i = 1:num_classes
    TP = confusion_matrix(i, i); % True Positives
    FP = sum(confusion_matrix(:, i)) - TP; % False Positives
    FN = sum(confusion_matrix(i, :)) - TP; % False Negatives
    
    % Precision, Recall och F1
    precision(i) = TP / (TP + FP + eps);
    recall(i) = TP / (TP + FN + eps);
    f1_score(i) = 2 * (precision(i) * recall(i)) / (precision(i) + recall(i) + eps);
end

% Visa resultat
disp('Förvirringsmatris:');
disp(confusion_matrix);

disp('Precision, Recall och F1 Score för varje klass:');
for i = 1:num_classes
    fprintf('Klass %s: Precision = %.2f, Recall = %.2f, F1 Score = %.2f\n', ...
            class_labels{i}, precision(i), recall(i), f1_score(i));
end

% Visa klassmappning
disp('Klassmappning:');
for i = 1:num_classes
    fprintf('Klass %d: %s\n', i, class_labels{i});
end

% Funktioner
% Softmax-funktion
function probs = softmax(scores)
    exp_scores = exp(scores - max(scores, [], 2)); % Numerisk stabilitet
    probs = exp_scores ./ sum(exp_scores, 2);
end

% Kostnadsfunktion för Softmax Regression
function [cost, grad] = softmaxCostFunction(theta, features, labels, num_classes)
    [m, n] = size(features);
    W = reshape(theta(1:n*num_classes), n, num_classes);
    b = theta(n*num_classes+1:end)';
    
    % Linjära kombinationer
    scores = features * W + b;
    probs = softmax(scores);
    
    % One-hot-encode labels
    one_hot_labels = full(sparse(1:m, labels, 1, m, num_classes));
    
    % Kostnadsfunktion
    cost = -sum(sum(one_hot_labels .* log(probs))) / m;
    
    % Gradienter
    grad_W = -(features' * (one_hot_labels - probs)) / m;
    grad_b = -sum(one_hot_labels - probs, 1) / m;
    
    % Kombinera gradienter
    grad = [grad_W(:); grad_b(:)];
end
