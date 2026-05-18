% run_phase2_v3.m
%
% Phase 2v3  –  Multi-view hard-negative drone detection training pipeline.
%
% WHAT IT DOES
%   1.  Checks required folders and available base models.
%   2.  Builds the augmented dataset (filter views + hard negatives).
%   3.  Extracts log-mel features.
%   4.  Trains (or fine-tunes) one CNN on all 5 filter views simultaneously.
%   5.  Evaluates: overall accuracy, per-condition, per-view.
%   6.  Sweeps detection threshold (0.3–0.7).
%   7.  Compares with Phase 2v2 (and Phase 3 if available).
%   8.  Saves model + results.  Prints final summary.
%
% KEY DESIGN PRINCIPLE
%   ONE CNN, not 5.  The model is trained with all filter views mixed
%   into the training set.  Hard negatives (tank-alone, engine-alone, …)
%   appear explicitly as no_drone so the model learns the difference between
%   "drone + noise" and "noise alone".
%
% OUTPUT MODEL
%   models/drone_cnn_phase2_v3_multiview_hardnegatives.mat
%
% PC SAFETY
%   - GPU auto-detected (RTX 3070 or any CUDA device).
%   - quickTestMode trains on a small subset first.
%   - miniBatchSize halved automatically on GPU out-of-memory.
%   - Checkpoints saved to results/phase2_v3/checkpoints/.
%
% USAGE
%   cd E:\drone_detect
%   run src/run_phase2_v3.m
%
% REQUIRED TOOLBOXES
%   Deep Learning Toolbox, Signal Processing Toolbox
%   Audio Toolbox (optional but preferred for melSpectrogram)

clearvars; clc;

% ── Paths ─────────────────────────────────────────────────────────────────
ROOT        = fileparts(fileparts(mfilename('fullpath')));
SRC_DIR     = fullfile(ROOT, 'src');
MODELS_DIR  = fullfile(ROOT, 'models');
RESULTS_DIR = fullfile(ROOT, 'results', 'phase2_v3');
FEAT_DIR    = fullfile(ROOT, 'features', 'phase2_v3');
CKPT_DIR    = fullfile(RESULTS_DIR, 'checkpoints');

addpath(SRC_DIR);

% Create directories
for d = {MODELS_DIR, RESULTS_DIR, FEAT_DIR, CKPT_DIR, ...
         fullfile(ROOT,'data','noise','tank'), ...
         fullfile(ROOT,'data','noise','engine'), ...
         fullfile(ROOT,'data','noise','wind'), ...
         fullfile(ROOT,'data','noise','traffic'), ...
         fullfile(ROOT,'data','noise','speech'), ...
         fullfile(ROOT,'data','noise','crowd'), ...
         fullfile(ROOT,'data','noise','custom')}
    if ~exist(d{1},'dir'), mkdir(d{1}); end
end

% ═══════════════════════════════════════════════════════════════════════════
%  CONFIGURATION  –  Edit these settings before running
% ═══════════════════════════════════════════════════════════════════════════
config = struct();

% GPU
config.useGPU             = true;     % use RTX 3070 if available

% Memory / dataset size
config.miniBatchSize      = 32;       % reduce to 16 or 8 if GPU OOM
config.maxExamplesPerClass= 10000;    % cap per class in full training
config.snrLevels          = [-20,-15,-10,-5,0,5,10];

% Quick test mode  (run this FIRST to verify the full pipeline quickly)
config.quickTestMode              = false;   % set true for a fast dry run
config.quickTestExamplesPerClass  = 500;

% Feature caching
config.saveIntermediateFeatures = true;

% Training
config.maxEpochs   = 50;
config.initLR      = 0.001;
config.valPatience = 5;

% ═══════════════════════════════════════════════════════════════════════════
fprintf('\n');
fprintf('╔══════════════════════════════════════════════════════════════╗\n');
fprintf('║   PHASE 2v3  –  Multi-view Hard-Negative Drone CNN          ║\n');
fprintf('╚══════════════════════════════════════════════════════════════╝\n\n');
fprintf('  Root dir    : %s\n', ROOT);
fprintf('  Quick mode  : %s\n', iif(config.quickTestMode,'YES (500/class)','NO (full)'));
fprintf('  GPU enabled : %s\n', iif(config.useGPU,'YES','NO'));
fprintf('  MaxEpochs   : %d\n', config.maxEpochs);
fprintf('  MiniBatch   : %d\n', config.miniBatchSize);
fprintf('  MaxExamples : %d / class\n', config.maxExamplesPerClass);
fprintf('\n');

% ── Step 1: Check folder structure ────────────────────────────────────────
fprintf('── Step 1: Checking folders ──────────────────────────────────\n');
DRONE_DIR = fullfile(ROOT,'data','raw','drone');
if ~exist(DRONE_DIR,'dir') || isempty(dir(fullfile(DRONE_DIR,'*.wav')))
    error(['run_phase2_v3: no drone WAV files found in:\n  %s\n' ...
           'Place drone WAV files there first.'], DRONE_DIR);
end
nDroneFiles = numel(dir(fullfile(DRONE_DIR,'*.wav')));
fprintf('  Drone files : %d in %s\n', nDroneFiles, DRONE_DIR);

noiseTypes = {'tank','engine','wind','traffic','speech','crowd','custom'};
for nt = 1:numel(noiseTypes)
    nd = fullfile(ROOT,'data','noise',noiseTypes{nt});
    nf = numel(dir(fullfile(nd,'*.wav')));
    if nf > 0
        fprintf('  Noise %-8s: %d WAV files\n', noiseTypes{nt}, nf);
    else
        fprintf('  Noise %-8s: 0 files  (will use synthetic fallback)\n', noiseTypes{nt});
    end
end

% Check available base models for fine-tuning
fprintf('\n  Available base models:\n');
modelCandidates = { ...
    'drone_cnn_phase2_v2_noise_speech_robust.mat', ...
    'drone_cnn_phase1.mat' ...
};
for mi = 1:numel(modelCandidates)
    mp = fullfile(MODELS_DIR, modelCandidates{mi});
    if isfile(mp)
        fprintf('    [FOUND]   %s\n', modelCandidates{mi});
    else
        fprintf('    [missing] %s\n', modelCandidates{mi});
    end
end

SAVE_PATH = fullfile(MODELS_DIR,'drone_cnn_phase2_v3_multiview_hardnegatives.mat');
if isfile(SAVE_PATH)
    fprintf('\n  WARNING: output model already exists:\n  %s\n', SAVE_PATH);
    fprintf('  It will be OVERWRITTEN.\n');
end
fprintf('\n');

% ── Step 2: GPU detection ─────────────────────────────────────────────────
fprintf('── Step 2: GPU detection ─────────────────────────────────────\n');
gpuAvailable = false;
if config.useGPU
    try
        gpuInfo = gpuDevice(1);
        gpuAvailable = true;
        fprintf('  GPU detected : %s\n', gpuInfo.Name);
        fprintf('  VRAM total   : %.1f GB\n', gpuInfo.TotalMemory/1e9);
        fprintf('  VRAM free    : %.1f GB\n', gpuInfo.AvailableMemory/1e9);
    catch
        fprintf('  No GPU found → training on CPU (slower).\n');
    end
else
    fprintf('  GPU disabled in config.\n');
end
fprintf('\n');

% ── Step 3: Build dataset ─────────────────────────────────────────────────
fprintf('── Step 3: Building Phase 2v3 dataset ───────────────────────\n');
t_data = tic;
[XTrain, YTrain, XVal, YVal, XTest, YTest, meta] = ...
    create_phase2_v3_dataset(config, ROOT);
fprintf('  Dataset built in %.1f s\n', toc(t_data));

nTrDrone   = sum(YTrain=='drone');
nTrNodrone = sum(YTrain=='no_drone');
nValTotal  = size(XVal,  4);
nTeTotal   = size(XTest, 4);

fprintf('\n  ┌─────────────────────────────────────────┐\n');
fprintf(  '  │  Dataset summary                        │\n');
fprintf(  '  ├─────────────────────────────────────────┤\n');
fprintf(  '  │  Feature size  : [%3d × %3d × 1]        │\n', meta.specH, meta.specW);
fprintf(  '  │  Train: drone  : %6d                  │\n', nTrDrone);
fprintf(  '  │  Train: nodrone: %6d                  │\n', nTrNodrone);
fprintf(  '  │  Validation    : %6d                  │\n', nValTotal);
fprintf(  '  │  Test          : %6d                  │\n', nTeTotal);
fprintf(  '  │  SNR levels    : %s        │\n', mat2str(meta.snrLevels));
fprintf(  '  └─────────────────────────────────────────┘\n\n');

% Estimate GPU memory requirement
nTotal       = size(XTrain,4);
featureBytes = nTotal * meta.specH * meta.specW * 4;  % single
fprintf('  Estimated feature RAM: %.1f MB\n\n', featureBytes/1e6);

% ── Step 4: Train CNN ─────────────────────────────────────────────────────
fprintf('── Step 4: Training Phase 2v3 CNN ───────────────────────────\n');
t_train = tic;
net = train_phase2_v3_cnn(XTrain, YTrain, XVal, YVal, config, ROOT);
fprintf('  Training completed in %.1f s  (%.1f min)\n\n', ...
        toc(t_train), toc(t_train)/60);

% ── Step 5: Overall evaluation ────────────────────────────────────────────
fprintf('── Step 5: Overall evaluation ────────────────────────────────\n');
metrics = evaluate_phase2_v3(net, XTest, YTest, RESULTS_DIR);

% ── Step 6: Per-condition evaluation ─────────────────────────────────────
fprintf('── Step 6: Per-condition evaluation ─────────────────────────\n');
condMetrics = evaluate_phase2_v3_by_condition(net, config, ROOT, RESULTS_DIR);

% ── Step 7: Per-view evaluation ───────────────────────────────────────────
fprintf('── Step 7: Per-view evaluation ───────────────────────────────\n');
viewMetrics = evaluate_phase2_v3_by_view(net, config, ROOT, RESULTS_DIR);

% ── Step 8: Threshold sweep ───────────────────────────────────────────────
fprintf('── Step 8: Threshold sweep ───────────────────────────────────\n');
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7];
droneIdx   = 1;
try
    ll = net.Layers(end);
    if isprop(ll,'Classes')
        cls = string(ll.Classes);
        idx = find(strcmpi(cls,'drone'),1);
        if ~isempty(idx), droneIdx = idx; end
    end
catch; end

% Compute multiview scores on test set
WEIGHTS = [0.05, 0.20, 0.25, 0.35, 0.15];
nTest   = size(XTest,4);
testScores = zeros(nTest, 1);
for si = 1:nTest
    win = double(XTest(:,:,1,si));  % already log-mel
    % For threshold sweep use the stored log-mel directly on each view
    % (here: just run single-view prediction on each filter view via re-extract)
    try
        X    = single(reshape(win, meta.specH, meta.specW, 1, 1));
        sc   = predict(net, X, 'MiniBatchSize', 1);
        testScores(si) = double(sc(droneIdx));
    catch
        testScores(si) = 0;
    end
end

fprintf('\n  %-8s  %-10s  %-12s  %-12s  %-12s\n', ...
        'Thresh', 'DroneRecall', 'TankFA', 'EngineFA', 'SpeechFA');
fprintf('  %s\n', repmat('-',1,60));

thrSweep = struct();
csvThrRows = {'threshold,drone_recall,tank_FA,engine_FA,speech_FA'};

for ti = 1:numel(THRESHOLDS)
    thr  = THRESHOLDS(ti);
    dets = testScores > thr;

    droneRecall = NaN; tankFA = NaN; engineFA = NaN; speechFA = NaN;

    % drone recall: from positive test windows
    droneIdxW = find(YTest == 'drone');
    if ~isempty(droneIdxW)
        droneRecall = mean(dets(droneIdxW)) * 100;
    end

    % condition-based FA from condMetrics
    if isfield(condMetrics,'tank_alone')
        tankFA   = mean(condMetrics.tank_alone.probs   > thr) * 100;
    end
    if isfield(condMetrics,'engine_alone')
        engineFA = mean(condMetrics.engine_alone.probs > thr) * 100;
    end
    if isfield(condMetrics,'speech_alone')
        speechFA = mean(condMetrics.speech_alone.probs > thr) * 100;
    end

    fprintf('  %-8.2f  %-10s  %-12s  %-12s  %-12s\n', thr, ...
        fmtPct(droneRecall), fmtPct(tankFA), fmtPct(engineFA), fmtPct(speechFA));

    fname = sprintf('thr_%02d', round(thr*10));
    thrSweep.(fname).threshold  = thr;
    thrSweep.(fname).droneRecall= droneRecall;
    thrSweep.(fname).tankFA     = tankFA;
    thrSweep.(fname).engineFA   = engineFA;
    thrSweep.(fname).speechFA   = speechFA;
    csvThrRows{end+1} = sprintf('%.2f,%.1f,%.1f,%.1f,%.1f', ...
        thr, nanDefault(droneRecall,NaN), nanDefault(tankFA,NaN), ...
        nanDefault(engineFA,NaN), nanDefault(speechFA,NaN)); %#ok<AGROW>
end

% Recommend threshold: highest recall where tankFA < 20%
recThr = NaN;
for ti = numel(THRESHOLDS):-1:1
    fname = sprintf('thr_%02d', round(THRESHOLDS(ti)*10));
    tf    = thrSweep.(fname).tankFA;
    dr    = thrSweep.(fname).droneRecall;
    if ~isnan(tf) && tf < 20 && ~isnan(dr)
        recThr = THRESHOLDS(ti);
        break;
    end
end
if ~isnan(recThr)
    fprintf('\n  Recommended threshold: %.2f  (tankFA < 20%% + best recall)\n', recThr);
else
    fprintf('\n  Recommended threshold: 0.50  (default)\n');
    recThr = 0.50;
end

save(fullfile(RESULTS_DIR,'threshold_sweep.mat'), 'thrSweep');
fid = fopen(fullfile(RESULTS_DIR,'threshold_sweep_table.csv'),'w');
for ri=1:numel(csvThrRows), fprintf(fid,'%s\n',csvThrRows{ri}); end
fclose(fid);
fprintf('  Saved: threshold_sweep.mat, threshold_sweep_table.csv\n\n');

% ── Step 9: Model comparison ──────────────────────────────────────────────
fprintf('── Step 9: Model comparison ──────────────────────────────────\n');
try
    compTable = compare_phase2v2_phase2v3(net, config, ROOT, RESULTS_DIR);
catch ME
    warning('run_phase2_v3: comparison failed: %s', ME.message);
end

% ── Step 10: Final summary ────────────────────────────────────────────────
fprintf('\n');
fprintf('╔══════════════════════════════════════════════════════════════╗\n');
fprintf('║   PHASE 2v3  –  TRAINING COMPLETE                           ║\n');
fprintf('╠══════════════════════════════════════════════════════════════╣\n');
fprintf('║  Model saved : drone_cnn_phase2_v3_multiview_hardnegatives  ║\n');
fprintf('║                                                              ║\n');
fprintf('║  Test set results:                                           ║\n');
fprintf('║    Accuracy  : %5.1f %%                                      ║\n', metrics.accuracy*100);
fprintf('║    Recall    : %5.1f %%                                      ║\n', metrics.recall*100);
fprintf('║    FP Rate   : %5.1f %%                                      ║\n', metrics.fpRate*100);
fprintf('║    F1-Score  : %5.1f %%                                      ║\n', metrics.f1*100);
fprintf('║                                                              ║\n');
fprintf('║  Key condition results (multiview, thr=0.5):                ║\n');

printCond = {'drone_alone','drone_tank_0dB','drone_tank_minus5dB', ...
             'drone_tank_minus10dB','tank_alone','engine_alone','speech_alone'};
for ci = 1:numel(printCond)
    cname = printCond{ci};
    if isfield(condMetrics, cname)
        cm = condMetrics.(cname);
        dr50 = mean(cm.probs > 0.5) * 100;
        lbl  = iif(cm.isPositive,'recall','FA   ');
        fprintf('║    %-28s  %s: %5.1f %%           ║\n', cname, lbl, dr50);
    end
end

fprintf('║                                                              ║\n');
fprintf('║  Recommended threshold : %.2f                              ║\n', recThr);
fprintf('║                                                              ║\n');
fprintf('║  Results saved to: results/phase2_v3/                       ║\n');
fprintf('╚══════════════════════════════════════════════════════════════╝\n\n');

fprintf('Done.\n');


% ─────────────────────────────────────────────────────────────────────────
%  Local helpers
% ─────────────────────────────────────────────────────────────────────────
function s = fmtPct(v)
    if isnan(v), s = '  N/A  '; else, s = sprintf('%5.1f %%', v); end
end

function v = nanDefault(x, def)
    if isnan(x), v = def; else, v = x; end
end

function out = iif(cond, a, b)
    if cond, out = a; else, out = b; end
end
