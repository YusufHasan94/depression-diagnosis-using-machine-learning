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

% Load and preprocess the data (assuming X_train and X_test are 2D matrices)

inputSize = size(X_train, 2);  % Input size for 1D CNN

% Define the 1D CNN architecture
layers = [
    sequenceInputLayer(inputSize)
    convolution1dLayer(3, 16, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling1dLayer(2, 'Stride', 2)
    convolution1dLayer(3, 32, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling1dLayer(2, 'Stride', 2)
    fullyConnectedLayer(64)
    reluLayer
    fullyConnectedLayer(numel(unique(y_train)))
    softmaxLayer
    classificationLayer
];

% Set training options
options = trainingOptions('sgdm', 'MaxEpochs', 10, 'Verbose', true, ...
    'Plots', 'training-progress');

% Convert the labels to categorical
y_train_categorical = categorical(y_train);
y_test_categorical = categorical(y_test);

% Train the 1D CNN model
net = trainNetwork(X_train, y_train_categorical, layers, options);

% Make predictions on the test set
y_pred_cnn = classify(net, X_test);

% Convert predicted labels to numeric values for comparison
y_pred_cnn_numeric = grp2idx(y_pred_cnn);

% Model evaluation
accuracy_cnn = sum(y_pred_cnn_numeric == y_test) / numel(y_test);
fprintf('1D CNN Accuracy is %.2f\n', accuracy_cnn * 100);

% Confusion matrix and visualization
C_cnn = confusionmat(y_test, y_pred_cnn_numeric);
fprintf('1D CNN Confusion matrix:\n');
disp(C_cnn);

% Plot confusion matrix
figure;
confusionchart(y_test, y_pred_cnn_numeric);
title('1D CNN Confusion Matrix');


