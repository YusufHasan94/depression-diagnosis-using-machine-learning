data = readtable('EEG.machinelearing_data_BRMH.csv');
data(:, {'no_', 'age', 'eeg_date', 'education', 'IQ', 'sex'}) = [];
data.Properties.VariableNames{'main_disorder'} = 'main_disorder';
data.Properties.VariableNames{'specific_disorder'} = 'specific_disorder';
features_with_null = data.Properties.VariableNames(sum(ismissing(data), 1) > 0);
data(:, features_with_null) = [];
main_disorders = unique(data.main_disorder);
specific_disoders = unique(data.specific_disorder);
mood_data = data(strcmp(data.main_disorder, 'Mood disorder') | ...
    strcmp(data.main_disorder, 'Healthy control'),:);
specific_disoders_encoding = grp2idx(mood_data.main_disorder);
features = table2array(mood_data(:, setdiff(mood_data.Properties.VariableNames,...
   {'main_disorder', 'specific_disorder'})));

delta_cols = mood_data(:, contains(mood_data.Properties.VariableNames, 'delta')).Variables;
beta_cols = mood_data(:, contains(mood_data.Properties.VariableNames, 'beta')).Variables;
theta_cols = mood_data(:, contains(mood_data.Properties.VariableNames, 'theta')).Variables;
alpha_cols = mood_data(:, contains(mood_data.Properties.VariableNames, 'alpha')).Variables;

req_features = [delta_cols, beta_cols, theta_cols, alpha_cols,];
X = zscore(req_features);
y = specific_disoders_encoding;

rng default;
total_samples = size(X, 1);
idx = randperm(total_samples);
train_ratio = .7;
train_samples = floor(train_ratio * total_samples);

X_train = X(idx(1:train_samples), :);
X_test = X(idx(train_samples+1:end), :);
y_train = y(idx(1:train_samples), :);
y_test = y(idx(train_samples+1:end), :);

% Train the TreeBagger model
numTrees = 100;
model = TreeBagger(numTrees, X_train, y_train, 'Method', 'classification');

y_pred_xg_boost = predict(model, X_test);

% Convert predicted labels to integers for comparison
y_pred_xg_boost = str2double(y_pred_xg_boost);

% Model evaluation
accuracy_xg_boost  = sum(y_pred_xg_boost == y_test) / numel(y_test);
fprintf('XGBoost Accuracy is %.2f\n', accuracy_xg_boost * 100);

% Confusion matrix and visualization
C_xg_boost = confusionmat(y_test, y_pred_xg_boost);
fprintf('XGBoost Confusion matrix:\n');
disp(C_xg_boost);

% Plot confusion matrix
figure;
confusionchart(y_test, y_pred_xg_boost);
title('XGBoost Confusion Matrix');


% Get predicted class probabilities for positive class
[~, scores_xg_boost] = predict(model, X_test);

% Extract the positive class probability (class 2)
scores_xg_boost_pos = scores_xg_boost(:, 2);

% Create an ROC curve
[Xg_boost_fpr, Xg_boost_tpr, ~, Xg_boost_auc] = perfcurve(y_test, scores_xg_boost_pos, 2);

% Plot ROC curve
figure;
plot(Xg_boost_fpr, Xg_boost_tpr, 'b-', 'LineWidth', 2);
hold on;

% Plot the random guess line
plot([0, 1], [0, 1], 'k--', 'LineWidth', 2);

xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('ROC Curve - XGBoost');
legend(['AUC = ', num2str(Xg_boost_auc)], 'Random Guess', 'Location', 'Best');
grid on;
hold off;


