function net = train_phase2_v3_cnn(XTrain, YTrain, XVal, YVal, config, rootDir)
% TRAIN_PHASE2_V3_CNN  Train (or fine-tune) the Phase 2v3 DroneCNN.
%
%   net = train_phase2_v3_cnn(XTrain, YTrain, XVal, YVal, config, rootDir)
%
%   INPUT
%     XTrain/XVal  [H × W × 1 × N] single feature arrays
%     YTrain/YVal  [N × 1] categorical arrays  ({'drone','no_drone'})
%     config       training configuration struct
%     rootDir      project root path
%
%   OUTPUT
%     net          trained DAGNetwork / SeriesNetwork
%
%   FINE-TUNING PRIORITY
%     1. models/drone_cnn_phase2_v2_noise_speech_robust.mat
%     2. models/drone_cnn_phase1.mat
%     3. Train from scratch
%
%   SAVED TO
%     models/drone_cnn_phase2_v3_multiview_hardnegatives.mat
%
%   PC SAFETY
%     - GPU training only if RTX 3070 (or any GPU) is detected.
%     - Falls back to CPU automatically.
%     - Checkpoints saved every N epochs.
%     - mini-batch size configurable; on GPU OOM error, halved automatically.

% ── Config defaults ───────────────────────────────────────────────────────
if ~isfield(config, 'useGPU'),         config.useGPU = true; end
if ~isfield(config, 'miniBatchSize'),  config.miniBatchSize = 32; end
if ~isfield(config, 'maxEpochs'),      config.maxEpochs = 50; end
if ~isfield(config, 'initLR'),         config.initLR = 0.001; end
if ~isfield(config, 'valPatience'),    config.valPatience = 5; end

MODELS_DIR   = fullfile(rootDir, 'models');
CKPT_DIR     = fullfile(rootDir, 'results', 'phase2_v3', 'checkpoints');
SAVE_PATH    = fullfile(MODELS_DIR, 'drone_cnn_phase2_v3_multiview_hardnegatives.mat');

if ~exist(CKPT_DIR,  'dir'), mkdir(CKPT_DIR);  end
if ~exist(MODELS_DIR,'dir'), mkdir(MODELS_DIR); end

% ── Feature map size ──────────────────────────────────────────────────────
SPEC_H = size(XTrain, 1);
SPEC_W = size(XTrain, 2);
inputSize  = [SPEC_H, SPEC_W, 1];
numClasses = numel(categories(YTrain));
fprintf('[Train] Input [%d × %d × 1]   Classes: %d   Samples: %d\n', ...
        SPEC_H, SPEC_W, numClasses, size(XTrain,4));

% ── GPU detection ─────────────────────────────────────────────────────────
useGPU = false;
if config.useGPU
    try
        gpu = gpuDevice(1);
        useGPU = true;
        fprintf('[Train] GPU detected: %s  (%.1f GB VRAM)\n', ...
                gpu.Name, gpu.TotalMemory / 1e9);
    catch
        fprintf('[Train] No GPU found → training on CPU.\n');
    end
end
executionEnv = iif(useGPU, 'gpu', 'auto');

% ── Load pretrained base model (if available) ─────────────────────────────
MODEL_PRIORITY = { ...
    'drone_cnn_phase2_v2_noise_speech_robust.mat', ...
    'drone_cnn_phase1.mat' ...
};
baseNet  = [];
baseName = '';
initLR   = config.initLR;

for mi = 1:numel(MODEL_PRIORITY)
    mpath = fullfile(MODELS_DIR, MODEL_PRIORITY{mi});
    if isfile(mpath)
        fprintf('[Train] Attempting to load base model: %s\n', MODEL_PRIORITY{mi});
        try
            tmp = load(mpath);
            fnames = fieldnames(tmp);
            for fi = 1:numel(fnames)
                cand = tmp.(fnames{fi});
                if isa(cand,'SeriesNetwork') || isa(cand,'DAGNetwork') || isa(cand,'dlnetwork')
                    baseNet  = cand;
                    baseName = MODEL_PRIORITY{mi};
                    break;
                end
            end
            if ~isempty(baseNet)
                fprintf('[Train] Using %s as starting point (LR = %.5f)\n', baseName, initLR*0.1);
                initLR = initLR * 0.1;   % 10× lower LR for fine-tuning
                break;
            end
        catch ME
            fprintf('[Train] Could not load %s: %s\n', MODEL_PRIORITY{mi}, ME.message);
        end
    end
end

if isempty(baseNet)
    fprintf('[Train] No base model found → training from scratch.\n');
end

% ── Build layer graph ─────────────────────────────────────────────────────
if ~isempty(baseNet)
    % Fine-tune from existing network
    try
        lgraph = layerGraph(baseNet);
        % Set lower LR for convolutional layers; higher for FC
        layers = lgraph.Layers;
        for li = 1:numel(layers)
            lyr = layers(li);
            if isa(lyr, 'nnet.cnn.layer.Convolution2DLayer')
                lyr = setLearnRateFactor(lyr, 'Weights', 0.5);
                lyr = setLearnRateFactor(lyr, 'Bias',    0.5);
                lgraph = replaceLayer(lgraph, lyr.Name, lyr);
            elseif isa(lyr, 'nnet.cnn.layer.FullyConnectedLayer')
                lyr = setLearnRateFactor(lyr, 'Weights', 2);
                lyr = setLearnRateFactor(lyr, 'Bias',    2);
                lgraph = replaceLayer(lgraph, lyr.Name, lyr);
            end
        end
        fprintf('[Train] Fine-tuning existing network.\n');
    catch ME
        fprintf('[Train] Layer graph modification failed (%s) → scratch.\n', ME.message);
        lgraph = build_drone_cnn(inputSize, numClasses);
    end
else
    lgraph = build_drone_cnn(inputSize, numClasses);
    fprintf('[Train] Built DroneCNN from scratch.\n');
end

% Print architecture summary
try
    analyzeNetwork(lgraph);
catch
    % analyzeNetwork may open GUI; skip if running headless
end

% ── Training options ──────────────────────────────────────────────────────
batchSize = config.miniBatchSize;

options = trainingOptions('adam', ...
    'MaxEpochs',            config.maxEpochs, ...
    'MiniBatchSize',        batchSize, ...
    'InitialLearnRate',     initLR, ...
    'LearnRateSchedule',    'piecewise', ...
    'LearnRateDropPeriod',  15, ...
    'LearnRateDropFactor',  0.3, ...
    'L2Regularization',     1e-4, ...
    'ValidationData',       {XVal, YVal}, ...
    'ValidationFrequency',  max(1, floor(size(XTrain,4)/batchSize/2)), ...
    'ValidationPatience',   config.valPatience, ...
    'ExecutionEnvironment', executionEnv, ...
    'CheckpointPath',       CKPT_DIR, ...
    'Shuffle',              'every-epoch', ...
    'Plots',                'none', ...
    'Verbose',              true, ...
    'VerboseFrequency',     max(1, floor(size(XTrain,4)/batchSize/4)));

% ── Train with automatic batch-size halving on OOM ────────────────────────
trained = false;
while ~trained && batchSize >= 4
    try
        fprintf('[Train] Starting training  batch=%d  LR=%.5f  env=%s\n', ...
                batchSize, initLR, executionEnv);
        [net, trainInfo] = trainNetwork(XTrain, YTrain, lgraph, options); %#ok<ASGLU>
        trained = true;
    catch ME
        if contains(lower(ME.message), {'out of memory','memory','gpu'})
            batchSize = max(4, floor(batchSize/2));
            fprintf('[Train] GPU OOM → reducing miniBatchSize to %d\n', batchSize);
            options = trainingOptions('adam', ...
                'MaxEpochs',            config.maxEpochs, ...
                'MiniBatchSize',        batchSize, ...
                'InitialLearnRate',     initLR, ...
                'LearnRateSchedule',    'piecewise', ...
                'LearnRateDropPeriod',  15, ...
                'LearnRateDropFactor',  0.3, ...
                'L2Regularization',     1e-4, ...
                'ValidationData',       {XVal, YVal}, ...
                'ValidationFrequency',  max(1, floor(size(XTrain,4)/batchSize/2)), ...
                'ValidationPatience',   config.valPatience, ...
                'ExecutionEnvironment', executionEnv, ...
                'CheckpointPath',       CKPT_DIR, ...
                'Shuffle',              'every-epoch', ...
                'Plots',                'none', ...
                'Verbose',              true, ...
                'VerboseFrequency',     max(1, floor(size(XTrain,4)/batchSize/4)));
        else
            rethrow(ME);
        end
    end
end

if ~trained
    error('train_phase2_v3_cnn: training failed even at batch size 4.');
end

% ── Save ──────────────────────────────────────────────────────────────────
classes = categories(YTrain);
save(SAVE_PATH, 'net', 'classes', 'trainInfo');
fprintf('[Train] Model saved to:\n  %s\n', SAVE_PATH);

end  % main function


% =========================================================================
%  build_drone_cnn: construct layer graph from scratch
% =========================================================================
function lgraph = build_drone_cnn(inputSize, numClasses)
layers = [
    imageInputLayer(inputSize, 'Name', 'input', 'Normalization', 'none')

    convolution2dLayer(3, 16, 'Padding', 'same', 'Name', 'conv1')
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool1')

    convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv2')
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu2')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool2')

    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv3')
    batchNormalizationLayer('Name', 'bn3')
    reluLayer('Name', 'relu3')

    globalAveragePooling2dLayer('Name', 'gap')
    fullyConnectedLayer(numClasses, 'Name', 'fc', ...
        'WeightLearnRateFactor', 2, 'BiasLearnRateFactor', 2)
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output', ...
        'Classes', categorical({'drone','no_drone'}))
];
lgraph = layerGraph(layers);
end


% ── inline if helper ──────────────────────────────────────────────────────
function out = iif(cond, a, b)
if cond, out = a; else, out = b; end
end
