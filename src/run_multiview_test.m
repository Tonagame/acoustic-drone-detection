% run_multiview_test.m
%
% Multi-view drone detection test script.
%
% WHAT IT DOES
%   1. Loads the best available trained drone CNN model.
%   2. Reads all WAV files from   data/test_audio/
%   3. For each file:
%        - converts to mono, resamples to 16 kHz
%        - splits into 1-second windows with 50% overlap
%        - runs predict_drone_multiview on every window
%        - applies temporal smoothing: event = detected in ≥2 of last 3 windows
%        - stores per-window results
%   4. Saves all results to   results/multiview/multiview_results.mat
%   5. Prints a clean per-file summary table.
%   6. Saves a probability timeline plot to
%        results/multiview/multiview_timeline.png
%
% DETECTION RULE (sensitive – any path triggers)
%   filteredMax   > 0.75   (best filtered view)
%   OR weightedScore > 0.60
%   OR voteCount  >= 2     (at 0.55 threshold)
%
%   Temporal smoothing: drone EVENT declared if ≥2 of last 3 windows detected.
%
% REQUIRED TOOLBOXES
%   Deep Learning Toolbox, Audio Toolbox (or Signal Processing Toolbox)
%
% USAGE
%   Run from the drone_detect project root:
%       cd E:\drone_detect
%       run src/run_multiview_test.m

clearvars; clc;

% ── Paths ─────────────────────────────────────────────────────────────────
ROOT        = fileparts(fileparts(mfilename('fullpath')));  % project root
SRC_DIR     = fullfile(ROOT, 'src');
MODELS_DIR  = fullfile(ROOT, 'models');
TEST_DIR    = fullfile(ROOT, 'data', 'test_audio');
RESULTS_DIR = fullfile(ROOT, 'results', 'multiview');

addpath(SRC_DIR);
if ~exist(RESULTS_DIR, 'dir'), mkdir(RESULTS_DIR); end

% ── Constants ─────────────────────────────────────────────────────────────
FS_TARGET    = 16000;
WIN_SEC      = 1.0;
HOP_SEC      = 0.5;           % 50% overlap
WIN_SAMPLES  = round(WIN_SEC  * FS_TARGET);
HOP_SAMPLES  = round(HOP_SEC  * FS_TARGET);
SMOOTH_WINS  = 3;             % temporal smoothing window
SMOOTH_MIN   = 2;             % minimum detections in window to trigger event

% ── Load best available model ─────────────────────────────────────────────
MODEL_PRIORITY = { ...
    'drone_cnn_phase2_v2_noise_speech_robust.mat', ...
    'drone_cnn_phase2_noise_robust.mat',           ...
    'drone_cnn_phase1.mat'                         ...
};

net       = [];
modelUsed = '';

for mi = 1:numel(MODEL_PRIORITY)
    mpath = fullfile(MODELS_DIR, MODEL_PRIORITY{mi});
    if isfile(mpath)
        fprintf('Loading model: %s\n', MODEL_PRIORITY{mi});
        try
            tmp    = load(mpath);
            fnames = fieldnames(tmp);
            for fi = 1:numel(fnames)
                candidate = tmp.(fnames{fi});
                if isa(candidate, 'SeriesNetwork')  || ...
                   isa(candidate, 'DAGNetwork')     || ...
                   isa(candidate, 'dlnetwork')
                    net = candidate;
                    break;
                end
            end
            if ~isempty(net)
                modelUsed = MODEL_PRIORITY{mi};
                break;
            else
                warning('run_multiview_test: %s loaded but no network found inside.', ...
                        MODEL_PRIORITY{mi});
            end
        catch ME
            warning('run_multiview_test: could not load %s: %s', ...
                    MODEL_PRIORITY{mi}, ME.message);
        end
    end
end

if isempty(net)
    error(['run_multiview_test: no model file found in %s\n' ...
           'Expected one of:\n  %s\n  %s\n  %s\n'], ...
           MODELS_DIR, MODEL_PRIORITY{:});
end

fprintf('Model loaded: %s\n\n', modelUsed);

% ── Discover test WAV files ───────────────────────────────────────────────
if ~exist(TEST_DIR, 'dir')
    error(['run_multiview_test: test audio directory not found:\n  %s\n' ...
           'Create it and place WAV files there.'], TEST_DIR);
end

wavFiles = dir(fullfile(TEST_DIR, '*.wav'));
if isempty(wavFiles)
    error('run_multiview_test: no WAV files found in %s', TEST_DIR);
end

fprintf('Found %d WAV file(s) in %s\n\n', numel(wavFiles), TEST_DIR);
fprintf('%-38s  %7s  %7s  %7s  %7s  %8s  %8s  %s\n', ...
        'File', 'MaxFilt', 'MaxWgt', 'RawDet', 'SmoothEv', 'Reason', 'BestView', '');
fprintf('%s\n', repmat('-', 1, 95));

% ── Per-file storage ──────────────────────────────────────────────────────
allFileResults = struct();

for fi = 1:numel(wavFiles)
    fname = wavFiles(fi).name;
    fpath = fullfile(TEST_DIR, fname);

    % ── Read and prepare audio ────────────────────────────────────────────
    try
        [audio, fs] = audioread(fpath);
    catch ME
        warning('run_multiview_test: cannot read %s: %s', fname, ME.message);
        continue
    end

    if size(audio, 2) > 1
        audio = mean(audio, 2);
    end
    audio = double(audio(:));

    if fs ~= FS_TARGET
        audio = resample(audio, FS_TARGET, fs);
        fs    = FS_TARGET;
    end

    % ── Sliding-window inference ──────────────────────────────────────────
    nSamples  = length(audio);
    winStarts = 1 : HOP_SAMPLES : (nSamples - WIN_SAMPLES + 1);
    if isempty(winStarts)
        warning('run_multiview_test: %s is shorter than 1 second – skipping.', fname);
        continue
    end

    nWin           = numel(winStarts);
    windowTimes    = (winStarts - 1) / FS_TARGET;   % window start time (s)
    weightedScores = zeros(1, nWin);
    filteredMaxes  = zeros(1, nWin);
    voteCounts     = zeros(1, nWin);
    detectedFlags  = false(1, nWin);
    smoothedEvents = false(1, nWin);
    allProbs       = zeros(5, nWin);               % [nViews × nWindows]
    reasonLog      = cell(1,  nWin);

    % Temporal smoothing buffer (last SMOOTH_WINS window decisions)
    smoothBuffer   = false(1, SMOOTH_WINS);

    for wi = 1:nWin
        s      = winStarts(wi);
        window = audio(s : s + WIN_SAMPLES - 1);

        res = predict_drone_multiview(window, fs, net);

        weightedScores(wi) = res.weightedScore;
        filteredMaxes(wi)  = res.filteredMax;
        voteCounts(wi)     = res.voteCount;
        detectedFlags(wi)  = res.detected;
        allProbs(:, wi)    = res.probs(:);
        reasonLog{wi}      = res.debug.reason;

        % ── Temporal smoothing ────────────────────────────────────────────
        smoothBuffer        = [smoothBuffer(2:end), res.detected];
        smoothedEvents(wi)  = sum(smoothBuffer) >= SMOOTH_MIN;
    end

    % ── File-level summary ────────────────────────────────────────────────
    maxFilteredMax  = max(filteredMaxes);
    maxWeightedScore= max(weightedScores);
    nRawDetected    = sum(detectedFlags);
    nSmoothedEvents = sum(smoothedEvents);

    % Most common trigger reason (excluding notDetected)
    triggered = reasonLog(~strcmp(reasonLog, 'notDetected'));
    if ~isempty(triggered)
        reasonCounts = struct();
        for r = {'filteredMax','weightedScore','voteCount'}
            reasonCounts.(r{1}) = sum(strcmp(triggered, r{1}));
        end
        [~, bestR] = max([reasonCounts.filteredMax, ...
                          reasonCounts.weightedScore, ...
                          reasonCounts.voteCount]);
        rNames = {'filteredMax','weightedScore','voteCount'};
        mostCommonReason = rNames{bestR};
    else
        mostCommonReason = 'none';
    end

    % Best view (highest average probability across windows, filtered only)
    avgViewProbs = mean(allProbs, 2);
    [~, bestViewIdx] = max(avgViewProbs(2:5));   % among filtered views
    bestViewIdx  = bestViewIdx + 1;              % offset back to full index
    viewNames    = res.viewNames;
    bestViewName = viewNames{bestViewIdx};

    fprintf('%-38s  %7.4f  %7.4f  %7d  %8d  %8s  %s\n', ...
            fname, maxFilteredMax, maxWeightedScore, ...
            nRawDetected, nSmoothedEvents, mostCommonReason, bestViewName);

    % ── Store for saving ──────────────────────────────────────────────────
    safeKey = matlab.lang.makeValidName(fname);
    allFileResults.(safeKey).fileName          = fname;
    allFileResults.(safeKey).windowTimes       = windowTimes;
    allFileResults.(safeKey).weightedScores    = weightedScores;
    allFileResults.(safeKey).filteredMaxes     = filteredMaxes;
    allFileResults.(safeKey).voteCounts        = voteCounts;
    allFileResults.(safeKey).detectedFlags     = detectedFlags;
    allFileResults.(safeKey).smoothedEvents    = smoothedEvents;
    allFileResults.(safeKey).allProbs          = allProbs;
    allFileResults.(safeKey).viewNames         = viewNames;
    allFileResults.(safeKey).maxFilteredMax    = maxFilteredMax;
    allFileResults.(safeKey).maxWeightedScore  = maxWeightedScore;
    allFileResults.(safeKey).nRawDetected      = nRawDetected;
    allFileResults.(safeKey).nSmoothedEvents   = nSmoothedEvents;
    allFileResults.(safeKey).mostCommonReason  = mostCommonReason;
    allFileResults.(safeKey).bestViewName      = bestViewName;
    allFileResults.(safeKey).reasonLog         = reasonLog;
end

fprintf('%s\n\n', repmat('-', 1, 95));

% ── Save results ──────────────────────────────────────────────────────────
savePath = fullfile(RESULTS_DIR, 'multiview_results.mat');
save(savePath, 'allFileResults', 'modelUsed');
fprintf('Results saved to: %s\n', savePath);

% ── Timeline plot ─────────────────────────────────────────────────────────
fileKeys = fieldnames(allFileResults);
nFiles   = numel(fileKeys);

if nFiles == 0
    warning('run_multiview_test: no results to plot.');
else
    fig = figure('Name', 'Multi-view Drone Detection Timeline', ...
                 'Visible', 'off', ...
                 'Position', [100 100 1200 max(320, 280*nFiles)]);

    VIEW_COLORS = lines(5);   % one colour per spectral view

    for fi = 1:nFiles
        key = fileKeys{fi};
        r   = allFileResults.(key);
        t   = r.windowTimes;

        ax = subplot(nFiles, 1, fi);
        hold(ax, 'on');

        % ── Per-view probabilities (thin translucent lines) ───────────────
        for v = 1:5
            plot(ax, t, r.allProbs(v,:), ...
                 'Color', [VIEW_COLORS(v,:), 0.35], ...
                 'LineWidth', 0.8, ...
                 'DisplayName', r.viewNames{v});
        end

        % ── Weighted score (medium bold, blue) ────────────────────────────
        plot(ax, t, r.weightedScores, ...
             'b-', 'LineWidth', 1.6, 'DisplayName', 'Weighted score');

        % ── Filtered max (bold black) ─────────────────────────────────────
        plot(ax, t, r.filteredMaxes, ...
             'k-', 'LineWidth', 2.2, 'DisplayName', 'Filtered max');

        % ── Threshold lines ───────────────────────────────────────────────
        yline(ax, 0.75, 'r--', 'LineWidth', 1.2, ...
              'Label', 'filteredMax 0.75', ...
              'LabelVerticalAlignment', 'bottom', ...
              'DisplayName', 'filteredMax threshold');
        yline(ax, 0.60, 'm--', 'LineWidth', 1.0, ...
              'Label', 'score 0.60', ...
              'LabelVerticalAlignment', 'bottom', ...
              'DisplayName', 'score threshold');

        % ── Raw detections (light red shading) ───────────────────────────
        rawTimes = t(r.detectedFlags);
        for di = 1:numel(rawTimes)
            patch(ax, ...
                  [rawTimes(di), rawTimes(di)+HOP_SEC, ...
                   rawTimes(di)+HOP_SEC, rawTimes(di)], ...
                  [0, 0, 1, 1], ...
                  [1 0.3 0.3], 'FaceAlpha', 0.12, 'EdgeColor', 'none', ...
                  'HandleVisibility', 'off');
        end

        % ── Smoothed events (deeper red shading) ─────────────────────────
        smoothTimes = t(r.smoothedEvents);
        for di = 1:numel(smoothTimes)
            patch(ax, ...
                  [smoothTimes(di), smoothTimes(di)+HOP_SEC, ...
                   smoothTimes(di)+HOP_SEC, smoothTimes(di)], ...
                  [0, 0, 1, 1], ...
                  [0.85 0.1 0.1], 'FaceAlpha', 0.28, 'EdgeColor', 'none', ...
                  'HandleVisibility', 'off');
        end

        ylim(ax, [0 1]);
        if ~isempty(t)
            xlim(ax, [0, max(t) + HOP_SEC]);
        end
        ylabel(ax, 'Drone prob');
        xlabel(ax, 'Time (s)');
        title(ax, sprintf('%s  |  filtMax=%.3f  wgtScore=%.3f  rawDet=%d  events=%d  [%s]', ...
              r.fileName, r.maxFilteredMax, r.maxWeightedScore, ...
              r.nRawDetected, r.nSmoothedEvents, r.mostCommonReason), ...
              'Interpreter', 'none');

        if fi == 1
            legend(ax, 'Location', 'northeast', 'FontSize', 7, ...
                   'NumColumns', 2);
        end
        hold(ax, 'off');
    end

    sgtitle(fig, sprintf('Multi-view Drone Detection  [model: %s]', modelUsed), ...
            'FontSize', 11, 'FontWeight', 'bold', 'Interpreter', 'none');

    plotPath = fullfile(RESULTS_DIR, 'multiview_timeline.png');
    exportgraphics(fig, plotPath, 'Resolution', 150);
    close(fig);
    fprintf('Timeline plot saved to: %s\n', plotPath);
end

fprintf('\nDone.\n');
