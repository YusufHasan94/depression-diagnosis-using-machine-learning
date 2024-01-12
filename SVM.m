data = readtable('EEG.machinelearing_data_BRMH.csv');
data(:, {'no_', 'age', 'eeg_date', 'education', 'IQ', 'sex'}) = [];
data.Properties.VariableNames{'main_disorder'} = 'main_disorder';
data.Properties.VariableNames{'specific_disorder'} = 'specific_disorder';
features_with_null = data.Properties.VariableNames(sum(ismissing(data), 1) > 0);
data(:, features_with_null) = [];
main_disorders = unique(data.main_disorder);
specific_disoders = unique(data.specific_disorder);
mood_data = data(strcmp(data.specific_disorder, 'Depressive disorder') | ...
                 strcmp(data.specific_disorder, 'Healthy control'), :);
specific_disoders_encoding = grp2idx(mood_data.specific_disorder);
features = table2array(mood_data(:, setdiff(mood_data.Properties.VariableNames,...
    {'main_disorder', 'specific_disorder'})));

delta_cols = mood_data(:, contains(mood_data.Properties.VariableNames, 'delta')).Variables;
beta_cols = mood_data(:, contains(mood_data.Properties.VariableNames, 'beta')).Variables;
theta_cols = mood_data(:, contains(mood_data.Properties.VariableNames, 'theta')).Variables;
alpha_cols = mood_data(:, contains(mood_data.Properties.VariableNames, 'alpha')).Variables;

req_features = [delta_cols, beta_cols, theta_cols, alpha_cols];
X = zscore(req_features);
y = specific_disoders_encoding;

rng default;
total_samples = size(X, 1);
idx = randperm(total_samples);
train_ratio = 0.7;
train_samples = floor(train_ratio * total_samples);

X_train = X(idx(1:train_samples), :);
X_test = X(idx(train_samples+1:end), :);
y_train = y(idx(1:train_samples), :);
y_test = y(idx(train_samples+1:end), :);

% SVM classifier
svm_model = fitcsvm(X_train, y_train, 'KernelFunction', 'linear');  % Linear kernel as a starting point
y_pred = predict(svm_model, X_test);

% Model evaluation
accuracy = sum(y_pred == y_test) / numel(y_test);
fprintf('Accuracy is %.2f\n', accuracy*100);

% Classification report and confusion matrix (same as before)
C = confusionmat(y_test, y_pred);
fprintf('Confusion matrix');
disp(C);

figure;
confusionchart(y_test, y_pred);
title('Confusion Matrix');

%calculate mean
mean_accuracy = mean(accuracy);
fprintf('Mean score %.4f\n', mean_accuracy);

% Calculate standard deviation of accuracy
accuracies = zeros(1, 100);
for i = 1:100 
    idx = randperm(total_samples);
    train_samples = floor(train_ratio * total_samples);

    X_train = X(idx(1:train_samples), :);
    X_test = X(idx(train_samples+1:end), :);
    y_train = y(idx(1:train_samples), :);
    y_test = y(idx(train_samples+1:end), :);

    svm_model = fitcsvm(X_train, y_train, 'KernelFunction', 'linear');
    y_pred = predict(svm_model, X_test);

    accuracies(i) = sum(y_pred == y_test) / numel(y_test);
end

std_dev_accuracy = std(accuracies);
fprintf('Standard Deviation score %.4f\n', std_dev_accuracy);


% F1 Score, Precision, and Recall Calculations
num_classes = size(C, 1);

precision = zeros(num_classes, 1);
recall = zeros(num_classes, 1);
f1_score = zeros(num_classes, 1);

for i = 1:num_classes
    precision(i) = C(i,i) / sum(C(:, i));
    recall(i) = C(i,i) / sum(C(i, :));
    f1_score(i) = 2 * (precision(i) * recall(i)) / (precision(i) + recall(i));
end

macro_precision = mean(precision);
macro_recall = mean(recall);
macro_f1_score = mean(f1_score);

% Micro F1 Score Calculation
total_true_positives = sum(diag(C));
total_predicted_positives = sum(C, 1);
total_actual_positives = sum(C, 2);

micro_precision = total_true_positives / sum(total_predicted_positives);
micro_recall = total_true_positives / sum(total_actual_positives);
micro_f1_score = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall);

fprintf('Micro Precision Score: %.4f\n', micro_precision);
fprintf('Micro Recall Score: %.4f\n', micro_recall);
fprintf('Micro F1 Score: %.4f\n', micro_f1_score);

fprintf('Macro Precision Score: %.4f\n', macro_precision);
fprintf('Macro Recall Score: %.4f\n', macro_recall);
fprintf('Macro F1 Score: %.4f\n', macro_f1_score);




