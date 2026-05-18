function net = train_drone_cnn(trainFeatures, valFeatures, modelPath)
% TRAIN_DRONE_CNN  Build and train a small CNN for drone audio classification.
%
%   net = train_drone_cnn(trainFeatures, valFeatures, modelPath)
%
%   Inputs:
%     trainFeatures.X      - [H x W x 1 x N_train] single array
%     trainFeatures.labels - N_train x 1 categorical
%     valFeatures          - same structure, validation split
%     modelPath            - full path to save the .mat file
%
%   Output:
%     net - trained SeriesNetwork
%
%   Requires: Deep Learning Toolbox

[H, W, ~, ~] = size(trainFeatures.X);
fprintf('Input feature map: %d x %d x 1\n', H, W);

% ------------------------------------------------------------------
% Inverse-frequency class weights  (handles drone/no_drone imbalance)
% ------------------------------------------------------------------
classNames = categories(trainFeatures.labels);
nClasses   = numel(classNames);
nTotal     = numel(trainFeatures.labels);
classWeights = zeros(1, nClasses);
fprintf('Class distribution in training set:\n');
for k = 1 : nClasses
    nk               = sum(trainFeatures.labels == classNames{k});
    classWeights(k)  = nTotal / (nClasses * nk);   % inverse frequency
    fprintf('  %-12s  %6d samples  weight = %.4f\n', ...
        classNames{k}, nk, classWeights(k));
end

% ------------------------------------------------------------------
% Architecture
% ------------------------------------------------------------------
layers = [
    imageInputLayer([H W 1], ...
        'Name',          'input', ...
        'Normalization', 'zscore')

    % Block 1
    convolution2dLayer([3 3], 16, 'Padding', 'same', 'Name', 'conv1')
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1')
    maxPooling2dLayer([2 2], 'Stride', [2 2], 'Name', 'pool1')

    % Block 2
    convolution2dLayer([3 3], 32, 'Padding', 'same', 'Name', 'conv2')
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu2')
    maxPooling2dLayer([2 2], 'Stride', [2 2], 'Name', 'pool2')

    % Block 3
    convolution2dLayer([3 3], 64, 'Padding', 'same', 'Name', 'conv3')
    batchNormalizationLayer('Name', 'bn3')
    reluLayer('Name', 'relu3')
    globalAveragePooling2dLayer('Name', 'gap')

    fullyConnectedLayer(2,  'Name', 'fc')
    softmaxLayer('Name',    'softmax')
    classificationLayer('Name', 'output', ...
        'Classes',      classNames, ...
        'ClassWeights', classWeights)
];

% ------------------------------------------------------------------
% Training options
% ------------------------------------------------------------------
options = trainingOptions('adam', ...
    'MaxEpochs',           30, ...
    'MiniBatchSize',       64, ...
    'InitialLearnRate',    1e-3, ...
    'LearnRateSchedule',   'piecewise', ...
    'LearnRateDropFactor', 0.5, ...
    'LearnRateDropPeriod', 10, ...
    'ValidationData',      {valFeatures.X, valFeatures.labels}, ...
    'ValidationFrequency', 50, ...
    'Shuffle',             'every-epoch', ...
    'Verbose',             true, ...
    'Plots',               'training-progress', ...
    'ExecutionEnvironment','auto');

% ------------------------------------------------------------------
% Train
% ------------------------------------------------------------------
fprintf('Starting training (MaxEpochs=30, MiniBatchSize=64)...\n');
net = trainNetwork(trainFeatures.X, trainFeatures.labels, layers, options);

% ------------------------------------------------------------------
% Save
% ------------------------------------------------------------------
modelDir = fileparts(modelPath);
if ~isempty(modelDir) && ~isfolder(modelDir)
    mkdir(modelDir);
end
save(modelPath, 'net');
fprintf('Trained model saved: %s\n', modelPath);
end
