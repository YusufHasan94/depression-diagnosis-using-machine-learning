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


% Define perparameters
numEstimators = [100, 300, 500];
maxDepth = [1, 3, 6];
% Initialize variables to store best hyperparameters and accuracy
bestAccuracy = 0;
bestNumEstimators = 0;
bestMaxDepth = 0;

% Iterate over hyperparameter combinations
for i = 1:length(numEstimators)
    for j = 1:length(maxDepth)
        if isnan(maxDepth(j))
            randomForest = TreeBagger(numEstimators(i), X_train, y_train, 'Method', 'classification');
        else
            randomForest = TreeBagger(numEstimators(i), X_train, y_train, 'Method', 'classification', 'MaxNumSplits', maxDepth(j));
        end
        y_pred = predict(randomForest, X_test);
        y_pred_numeric = cellfun(@str2num, y_pred);
        
        accuracy = sum(y_pred_numeric == y_test) / numel(y_test);
        
        % Check if current hyperparameters yield better accuracy
        if accuracy > bestAccuracy
            bestAccuracy = accuracy;
            bestNumEstimators = numEstimators(i);
            bestMaxDepth = maxDepth(j);
        end
        
        % Display confusion matrix for the current hyperparameters
        fprintf('Confusion matrix for NumEstimators=%d, MaxDepth=%s:\n', numEstimators(i), num2str(maxDepth(j)));
        C = confusionmat(y_test, y_pred_numeric);
        disp(C);
        
        % Visualize confusion matrix as a heatmap
        figure;
        confusionchart(y_test, y_pred_numeric, 'Title', sprintf('Confusion Matrix - NumEstimators=%d, MaxDepth=%s', numEstimators(i), num2str(maxDepth(j))));       
    end
end


fprintf('Best Accuracy: %.2f%%\n', bestAccuracy * 100);
fprintf('Best Number of Estimators: %d\n', bestNumEstimators);
fprintf('Best Maximum Depth: %s\n', num2str(bestMaxDepth));

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







