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

% Reshape the feature data for CNN input
X_cnn = reshape(req_features, [size(req_features, 1), 1, 1, size(req_features, 2)]);

rng default;
total_samples = size(X_cnn, 1);
idx = randperm(total_samples);
train_ratio = 0.7;
train_samples = floor(train_ratio * total_samples);

X_train_cnn = X_cnn(idx(1:train_samples), :);
X_test_cnn = X_cnn(idx(train_samples+1:end), :);
y_train = y(idx(1:train_samples), :);
y_test = y(idx(train_samples+1:end), :);

% Reshape data for CNN input (assuming 2D structure for EEG signals)
X_train_cnn = reshape(X_train, [size(X_train, 1), size(X_train, 2), 1]);  % Add a channel dimension
X_test_cnn = reshape(X_test, [size(X_test, 1), size(X_test, 2), 1]);

% CNN architecture (adjust based on dataset and experimentation)
layers = [
    imageInputLayer([size(X_train_cnn, 2), size(X_train_cnn, 3)])
    
    convolution2dLayer(3, 8, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2, 'Stride', 2)
    
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer
];

% Create and train the CNN model
cnn_model = trainNetwork(X_train_cnn, y_train, layers, ...
    'adam', 'InitialLearnRate', 0.001, 'MaxEpochs', 10, 'MiniBatchSize', 32);

% Prediction
y_pred = classify(cnn_model, X_test_cnn);

% Model evaluation (same as before)
accuracy = sum(y_pred == y_test) / numel(y_test);
fprintf('Accuracy is %.2f\n', accuracy*100);

% Classification report and confusion matrix (same as before)
C = confusionmat(y_test, y_pred);
fprintf('Confusion matrix:\n');
disp(C);

figure;
confusionchart(y_test, y_pred, 'RowSummary', 'row-normalized', 'ColumnSummary', 'column-normalized');
title('Confusion Matrix');
