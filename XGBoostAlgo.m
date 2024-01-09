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


% Train the TreeBagger model
numTrees = 100;
model = TreeBagger(numTrees, X_train, y_train, 'Method', 'classification');

% Make predictions on the test set
y_pred_treebagger = predict(model, X_test);

% Convert predicted labels to integers for comparison
y_pred_treebagger = str2double(y_pred_treebagger);

% Model evaluation
accuracy_treebagger = sum(y_pred_treebagger == y_test) / numel(y_test);
fprintf('TreeBagger Accuracy is %.2f\n', accuracy_treebagger * 100);

% Confusion matrix and visualization
C_treebagger = confusionmat(y_test, y_pred_treebagger);
fprintf('TreeBagger Confusion matrix:\n');
disp(C_treebagger);

% Plot confusion matrix
figure;
confusionchart(y_test, y_pred_treebagger);
title('TreeBagger Confusion Matrix');



