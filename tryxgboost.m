data = readtable('EEG.machinelearing_data_BRMH.csv');
data(:, {'no_', 'age', 'eeg_date', 'education', 'IQ', 'sex'}) = [];
data.Properties.VariableNames{'main_disorder'} = 'main_disorder';
data.Properties.VariableNames{'specific_disorder'} = 'specific_disorder';
features_with_null = data.Properties.VariableNames(sum(ismissing(data), 1) > 0);
data(:, features_with_null) = [];
main_disorders = unique(data.main_disorder);
specific_disoders = unique(data.specific_disorder);
mood_data = data(strcmp(data.specific_disorder, 'Depressive disorder') | ... 
               strcmp(data.main_disorder, 'Healthy control'),:);
                 
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
% Define hyperparameters to search
numEstimators = [100, 300, 500];
subSample = [0.3, 0.5, 1];
maxDepth = [1, 3, 6, NaN];

% ... (existing code)

bestAccuracy = 0;
bestParams = [];

% Grid search
for numTrees = numEstimators
    for subsampleRatio = subSample
        for depth = maxDepth
            params = struct('NumTrees', numTrees, 'Method', 'classification');
            if subsampleRatio < 1
                params.SampleWithReplacement = true;
                params.FractionOfDataToSample = subsampleRatio;
            else
                params.SampleWithReplacement = false;
            end
            if ~isnan(depth)
                params.MaxNumSplits = 2^depth;
            else
                params.MaxNumSplits = NaN;
            end
            
            % Train the TreeBagger model
            model = TreeBagger(params.NumTrees, X_train, y_train, 'Method', params.Method, 'MaxNumSplits', params.MaxNumSplits);

            % Cross-validation
            %cv = cvpartition(size(X, 1), 'KFold', 5);
            %crossval_model = crossval(model, 'CVPartition', cv);
            %y_pred_xg_boost = kfoldPredict(crossval_model);

            % Convert predicted labels to integers for comparison
            y_pred_xg_boost = str2double(y_pred_xg_boost);

            % Model evaluation
            accuracy_xg_boost = sum(y_pred_xg_boost == y_test) / numel(y_test);
            
            % Update best parameters if accuracy is improved
            if accuracy_xg_boost > bestAccuracy
                bestAccuracy = accuracy_xg_boost;
                bestParams = params;
            end
        end
    end
end

% Train the final model with the best parameters
model = TreeBagger(bestParams.NumTrees, X_train, y_train, 'Method', bestParams.Method, 'FractionOfDataToSample', bestParams.FractionOfDataToSample, 'MaxNumSplits', bestParams.MaxNumSplits);

% Make predictions on the test set
y_pred_xg_boost = predict(model, X_test);

% Convert predicted labels to integers for comparison
y_pred_xg_boost = str2double(y_pred_xg_boost);

% Model evaluation
accuracy_xg_boost = sum(y_pred_xg_boost == y_test) / numel(y_test);
fprintf('Best XGBoost Accuracy is %.2f\n', accuracy_xg_boost * 100);

% Confusion matrix and visualization
C_xg_boost = confusionmat(y_test, y_pred_xg_boost);
fprintf('XGBoost Confusion matrix:\n');
disp(C_xg_boost);

% Plot confusion matrix
figure;
confusionchart(y_test, y_pred_xg_boost);
title('XGBoost Confusion Matrix');


