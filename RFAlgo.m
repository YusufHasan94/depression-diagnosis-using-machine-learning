data = readtable('EEG.machinelearing_data_BRMH.csv');
data(:, {'no_', 'age', 'eeg_date', 'education', 'IQ', 'sex'}) = [];
data.Properties.VariableNames{'main_disorder'} = 'main_disorder';
data.Properties.VariableNames{'specific_disorder'} = 'specific_disorder';
features_with_null = data.Properties.VariableNames(sum(ismissing(data), 1) > 0);
data(:, features_with_null) = [];
main_disorders = unique(data.main_disorder);
specific_disoders = unique(data.specific_disorder);
mood_data = data(data.main_disorder == "Mood disorder", :);

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


numTrees = 100;  % Number of trees in the forest
numFeatures = size(X_train, 2);  % Number of features

% Initialize cell arrays to store the trees and their predictions
trees = cell(numTrees, 1);
y_pred_random_forest = zeros(size(X_test, 1), numTrees);

for i = 1:numTrees
    % Bootstrap sampling with replacement for creating different subsets of the data
    idx = datasample(1:size(X_train, 1), size(X_train, 1));
    X_train_subset = X_train(idx, :);
    y_train_subset = y_train(idx);
    
    % Train individual decision trees
    tree = fitctree(X_train_subset, y_train_subset, 'NumVariablesToSample', numFeatures);
    
    % Store the trained tree
    trees{i} = tree;
    
    % Make predictions on the test set for each tree
    y_pred_random_forest(:, i) = predict(tree, X_test);
end

% Perform majority voting for the ensemble's predictions
y_pred_majority = mode(y_pred_random_forest, 2);

% Model evaluation
accuracy_random_forest = sum(y_pred_majority == y_test) / numel(y_test);
fprintf('Random Forest Accuracy is %.2f\n', accuracy_random_forest * 100);

% Confusion matrix and visualization
C_random_forest = confusionmat(y_test, y_pred_majority);
fprintf('Random Forest Confusion matrix:\n');
disp(C_random_forest);

% Plot confusion matrix
figure;
confusionchart(y_test, y_pred_majority);
title('Random Forest Confusion Matrix');



