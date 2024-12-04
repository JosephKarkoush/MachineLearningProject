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

% Skapa polynomiella funktioner av andra ordningen
poly_train_features = createPolynomialFeatures(train_features, 2);
poly_test_features = createPolynomialFeatures(test_features, 2);

% Standardisera funktionerna
mu = mean(poly_train_features);
sigma = std(poly_train_features);
poly_train_features = (poly_train_features - mu) ./ sigma;
poly_test_features = (poly_test_features - mu) ./ sigma;

% One-vs-rest Logistic Regression
models = cell(num_classes, 1); % Lagra en modell per klass
for i = 1:num_classes
    % Skapa binära etiketter för den aktuella klassen
    binary_labels = (train_labels_numeric == i);
    
    % Träna logistisk regression för den här klassen
    models{i} = fitclinear(poly_train_features, binary_labels, ...
                           'Learner', 'logistic', ...
                           'Regularization', 'ridge'); % Använd ridge-regularisering
end

% Förutsägelser på testdata
pred_scores = zeros(size(poly_test_features, 1), num_classes);
for i = 1:num_classes
    [~, score] = predict(models{i}, poly_test_features);
    pred_scores(:, i) = score(:, 2); % Sannolikheten för klass "1" för varje modell
end

% Bestäm förutspådd klass (max sannolikhet)
[~, predicted_classes] = max(pred_scores, [], 2);

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

% Funktion för att skapa polynomiella funktioner av andra ordningen
function poly_features = createPolynomialFeatures(features, degree)
    poly_features = features;
    num_features = size(features, 2);
    for i = 1:num_features
        for j = i:num_features
            poly_features = [poly_features, features(:, i) .* features(:, j)];
        end
    end
end
