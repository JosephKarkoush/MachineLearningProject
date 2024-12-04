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

% One-vs-rest Logistic Regression
models = cell(num_classes, 1); % Lagra en modell per klass
for i = 1:num_classes
    % Skapa binära etiketter för den aktuella klassen
    binary_labels = (train_labels_numeric == i);
    
    % Träna logistisk regression för den här klassen
    models{i} = fitclinear(train_features, binary_labels, ...
                           'Learner', 'logistic', ...
                           'Regularization', 'ridge'); % Använd ridge-regularisering
end

% Förutsägelser på testdata
pred_scores = zeros(size(test_features, 1), num_classes);
for i = 1:num_classes
    [~, score] = predict(models{i}, test_features);
    pred_scores(:, i) = score(:, 2); % Sannolikheten för klass "1" för varje modell
end

% Bestäm förutspådd klass (max sannolikhet)
[~, predicted_classes] = max(pred_scores, [], 2);

% Utvärdera prestanda
confusion_matrix = confusionmat(test_labels_numeric, predicted_classes);
accuracy = sum(diag(confusion_matrix)) / sum(confusion_matrix(:));
fprintf('Noggrannhet: %.2f%%\n', accuracy * 100);

% Visa förvirringsmatris
disp('Förvirringsmatris:');
disp(confusion_matrix);

% Visa klassmappning
disp('Klassmappning:');
for i = 1:num_classes
    fprintf('Klass %d: %s\n', i, class_labels{i});
end
