% RUN_PHASE1  Full pipeline: load data → extract features → train → evaluate.
%
%   Usage (from the project root directory):
%     run("src/run_phase1.m")
%
%   Prerequisites:
%     1. Place DADS WAV files in:
%          data/raw/drone/      (drone recordings)
%          data/raw/no_drone/   (background / no-drone recordings)
%     2. See README_phase1.md for download instructions.
%
%   Outputs:
%     models/drone_cnn_phase1.mat     trained network
%     features/logmel_features.mat    cached log-Mel feature arrays
%     results/phase1_metrics.mat      accuracy + per-class metrics
%     results/confusion_chart.png     confusion chart image

% ------------------------------------------------------------------
% Resolve paths relative to this script's location
% ------------------------------------------------------------------
scriptDir   = fileparts(mfilename('fullpath'));
projectRoot = fileparts(scriptDir);
addpath(scriptDir);   % make helper functions callable

dataRoot    = fullfile(projectRoot, 'data',     'raw');
featuresDir = fullfile(projectRoot, 'features');
modelPath   = fullfile(projectRoot, 'models',   'drone_cnn_phase1.mat');
resultsDir  = fullfile(projectRoot, 'results');

fprintf('================================================\n');
fprintf(' Phase 1  –  Drone Audio Detector\n');
fprintf('================================================\n');
fprintf('Project root : %s\n', projectRoot);
fprintf('Data root    : %s\n', dataRoot);

% ------------------------------------------------------------------
% Step 1: Prepare dataset  (load → preprocess → window)
% ------------------------------------------------------------------
fprintf('\n[1/4] Preparing dataset...\n');
t1 = tic;
[trainData, valData, testData] = prepare_dataset(dataRoot);
fprintf('  Done in %.1f s.\n', toc(t1));

% ------------------------------------------------------------------
% Step 2: Extract log-Mel features
% ------------------------------------------------------------------
fprintf('\n[2/4] Extracting log-Mel spectrograms...\n');
t2 = tic;

trainFeatures = extract_logmel(trainData);
valFeatures   = extract_logmel(valData);
testFeatures  = extract_logmel(testData);

fprintf('  Feature array shape: [%s] (train)\n', ...
    num2str(size(trainFeatures.X)));

% Cache features to disk (useful for re-running training without re-extraction)
if ~isfolder(featuresDir)
    mkdir(featuresDir);
end
featFile = fullfile(featuresDir, 'logmel_features.mat');
fprintf('  Saving features to %s ...\n', featFile);
save(featFile, 'trainFeatures', 'valFeatures', 'testFeatures', '-v7.3');
fprintf('  Done in %.1f s.\n', toc(t2));

% ------------------------------------------------------------------
% Step 3: Train CNN
% ------------------------------------------------------------------
fprintf('\n[3/4] Training CNN...\n');
t3 = tic;
net = train_drone_cnn(trainFeatures, valFeatures, modelPath);
fprintf('  Done in %.1f s.\n', toc(t3));

% ------------------------------------------------------------------
% Step 4: Evaluate on held-out test set
% ------------------------------------------------------------------
fprintf('\n[4/4] Evaluating on test set...\n');
t4 = tic;
metrics = evaluate_model(net, testFeatures, resultsDir);
fprintf('  Done in %.1f s.\n', toc(t4));

% ------------------------------------------------------------------
% Summary
% ------------------------------------------------------------------
fprintf('\n================================================\n');
fprintf(' Phase 1 complete!\n');
fprintf('   Test accuracy  : %.2f%%\n', metrics.accuracy * 100);
fprintf('   Model saved    : %s\n', modelPath);
fprintf('   Results saved  : %s\n', resultsDir);
fprintf('================================================\n');
