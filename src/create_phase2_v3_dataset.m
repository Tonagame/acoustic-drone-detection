function [XTrain, YTrain, XVal, YVal, XTest, YTest, meta] = ...
    create_phase2_v3_dataset(config, rootDir)
% CREATE_PHASE2_V3_DATASET  Build the multi-view hard-negative training dataset.
%
%   [XTrain,YTrain,XVal,YVal,XTest,YTest,meta] = create_phase2_v3_dataset(config,rootDir)
%
%   DESIGN
%     Each 1-second audio window generates up to 5 feature examples
%     (one per filter view: raw, HPF-150, HPF-250, BPF-200-6k, BPF-500-6k).
%     The CNN sees ALL views for EVERY label → filter-view invariant training.
%
%     HARD NEGATIVE PAIRS (mandatory when files exist):
%       drone + tank   → drone      tank alone     → no_drone
%       drone + engine → drone      engine alone   → no_drone
%       drone + speech → drone      speech alone   → no_drone
%       drone+tank+speech → drone   tank+speech    → no_drone
%
%   INPUT
%     config  - struct with fields:
%       .maxExamplesPerClass        max features per class total
%       .quickTestMode              if true, use .quickTestExamplesPerClass
%       .quickTestExamplesPerClass  small subset for pipeline testing
%       .saveIntermediateFeatures   save features to disk for reuse
%       .snrLevels                  vector of SNR dB values
%     rootDir - project root path string
%
%   OUTPUT
%     XTrain/XVal/XTest  [H × W × 1 × N] single arrays
%     YTrain/YVal/YTest  [N × 1] categorical arrays
%     meta               struct with dataset statistics

addpath(rootDir);

% ── Resolve directories ────────────────────────────────────────────────────
DATA_DIR   = fullfile(rootDir, 'data');
DRONE_DIR  = fullfile(DATA_DIR, 'raw', 'drone');
NODRONE_DIR= fullfile(DATA_DIR, 'raw', 'no_drone');
NOISE_BASE = fullfile(DATA_DIR, 'noise');
FEAT_DIR   = fullfile(rootDir, 'features', 'phase2_v3');
if ~exist(FEAT_DIR, 'dir'), mkdir(FEAT_DIR); end

% ── Config defaults ───────────────────────────────────────────────────────
if ~isfield(config, 'maxExamplesPerClass'),       config.maxExamplesPerClass = 10000; end
if ~isfield(config, 'quickTestMode'),             config.quickTestMode = false; end
if ~isfield(config, 'quickTestExamplesPerClass'), config.quickTestExamplesPerClass = 500; end
if ~isfield(config, 'saveIntermediateFeatures'),  config.saveIntermediateFeatures = true; end
if ~isfield(config, 'snrLevels'), config.snrLevels = [-20,-15,-10,-5,0,5,10]; end

maxPerClass = config.maxExamplesPerClass;
if config.quickTestMode
    maxPerClass = config.quickTestExamplesPerClass;
    fprintf('[Dataset] quickTestMode ON  →  %d examples/class\n', maxPerClass);
end

% ── Check for cached features ──────────────────────────────────────────────
cacheFile = fullfile(FEAT_DIR, sprintf('features_max%d%s.mat', maxPerClass, ...
    iif(config.quickTestMode, '_quick', '')));
if exist(cacheFile, 'file') && config.saveIntermediateFeatures
    fprintf('[Dataset] Loading cached features from:\n  %s\n', cacheFile);
    loaded = load(cacheFile);
    XTrain = loaded.XTrain; YTrain = loaded.YTrain;
    XVal   = loaded.XVal;   YVal   = loaded.YVal;
    XTest  = loaded.XTest;  YTest  = loaded.YTest;
    meta   = loaded.meta;
    fprintf('[Dataset] Loaded  Train=%d  Val=%d  Test=%d\n', ...
        size(XTrain,4), size(XVal,4), size(XTest,4));
    return;
end

FS_TARGET   = 16000;
WIN_SAMPLES = FS_TARGET;   % 1 second
HOP_SAMPLES = FS_TARGET / 2;
VIEW_NAMES  = {'raw','highpass_150Hz','highpass_250Hz', ...
               'bandpass_200_6000Hz','bandpass_500_6000Hz'};
N_VIEWS     = numel(VIEW_NAMES);
SNR_LEVELS  = config.snrLevels;

% ── Scan source files ──────────────────────────────────────────────────────
fprintf('[Dataset] Scanning source files ...\n');
droneFiles   = scan_wavs(DRONE_DIR);
nodroneFiles = scan_wavs(NODRONE_DIR);
noiseTypes   = {'tank','engine','wind','traffic','speech','crowd','custom'};
noiseFiles   = struct();
for nt = 1:numel(noiseTypes)
    noiseFiles.(noiseTypes{nt}) = scan_wavs(fullfile(NOISE_BASE, noiseTypes{nt}));
end

fprintf('  Drone files    : %d\n', numel(droneFiles));
fprintf('  No-drone files : %d\n', numel(nodroneFiles));
for nt = 1:numel(noiseTypes)
    fprintf('  Noise %-10s: %d files\n', noiseTypes{nt}, ...
            numel(noiseFiles.(noiseTypes{nt})));
end

if numel(droneFiles) == 0
    error('create_phase2_v3_dataset: no drone WAV files found in:\n  %s', DRONE_DIR);
end

% ── Determine feature map size ─────────────────────────────────────────────
testAudio = zeros(WIN_SAMPLES, 1);
testLM    = extract_logmel_phase2_v3(testAudio, FS_TARGET);
SPEC_H    = size(testLM, 1);
SPEC_W    = size(testLM, 2);
fprintf('[Dataset] Feature size: [%d × %d]\n', SPEC_H, SPEC_W);

% ── File-level train/val/test split ───────────────────────────────────────
rng(42);  % reproducible split
[drTr, drVa, drTe]   = split_files(droneFiles,   0.70, 0.15);
[ndTr, ndVa, ndTe]   = split_files(nodroneFiles, 0.70, 0.15);
noiseSplit = struct();
for nt = 1:numel(noiseTypes)
    flist = noiseFiles.(noiseTypes{nt});
    [noiseSplit.(noiseTypes{nt}).tr, ...
     noiseSplit.(noiseTypes{nt}).va, ...
     noiseSplit.(noiseTypes{nt}).te] = split_files(flist, 0.70, 0.15);
end

% ── Build each split ──────────────────────────────────────────────────────
fprintf('[Dataset] Building train split ...\n');
[Xd_tr, Xn_tr] = build_split(drTr, ndTr, noiseSplit, 'tr', ...
    maxPerClass, FS_TARGET, WIN_SAMPLES, HOP_SAMPLES, SNR_LEVELS, SPEC_H, SPEC_W);

fprintf('[Dataset] Building val split ...\n');
[Xd_va, Xn_va] = build_split(drVa, ndVa, noiseSplit, 'va', ...
    round(maxPerClass*0.15/0.70), FS_TARGET, WIN_SAMPLES, HOP_SAMPLES, SNR_LEVELS, SPEC_H, SPEC_W);

fprintf('[Dataset] Building test split ...\n');
[Xd_te, Xn_te] = build_split(drTe, ndTe, noiseSplit, 'te', ...
    round(maxPerClass*0.15/0.70), FS_TARGET, WIN_SAMPLES, HOP_SAMPLES, SNR_LEVELS, SPEC_H, SPEC_W);

% ── Assemble and label ────────────────────────────────────────────────────
classes = categorical({'drone','no_drone'});

[XTrain, YTrain] = assemble(Xd_tr, Xn_tr, maxPerClass);
[XVal,   YVal  ] = assemble(Xd_va, Xn_va, round(maxPerClass*0.15/0.70));
[XTest,  YTest ] = assemble(Xd_te, Xn_te, round(maxPerClass*0.15/0.70));

fprintf('[Dataset] Final sizes:\n');
fprintf('  Train  : drone=%d  no_drone=%d\n', sum(YTrain=='drone'), sum(YTrain=='no_drone'));
fprintf('  Val    : drone=%d  no_drone=%d\n', sum(YVal  =='drone'), sum(YVal  =='no_drone'));
fprintf('  Test   : drone=%d  no_drone=%d\n', sum(YTest =='drone'), sum(YTest =='no_drone'));

% ── Meta ──────────────────────────────────────────────────────────────────
meta.specH       = SPEC_H;
meta.specW       = SPEC_W;
meta.viewNames   = VIEW_NAMES;
meta.nDroneTrain = sum(YTrain=='drone');
meta.nNodroneTrain = sum(YTrain=='no_drone');
meta.snrLevels   = SNR_LEVELS;
meta.classes     = categories(YTrain);

% ── Cache ─────────────────────────────────────────────────────────────────
if config.saveIntermediateFeatures
    fprintf('[Dataset] Saving feature cache to:\n  %s\n', cacheFile);
    save(cacheFile, 'XTrain','YTrain','XVal','YVal','XTest','YTest','meta', '-v7.3');
end

end  % main function


% =========================================================================
%  build_split: drone + no_drone feature arrays for one data split
% =========================================================================
function [Xdrone, Xnodrone] = build_split(droneFiles, nodroneFiles, noiseSplit, ...
    splitKey, maxPerClass, FS, WIN, HOP, SNR_LEVELS, SPEC_H, SPEC_W)

N_VIEWS = 5;
% Pre-allocate (over-estimate; will trim later)
capacity = maxPerClass * 2;
Xdrone   = zeros(SPEC_H, SPEC_W, 1, capacity, 'single');
Xnodrone = zeros(SPEC_H, SPEC_W, 1, capacity, 'single');
nDrone   = 0;
nNodrone = 0;

noiseTypes = {'tank','engine','wind','traffic','speech','crowd','custom'};

% Collect noise windows for this split
noiseWins = struct();
for nt = 1:numel(noiseTypes)
    fld = noiseTypes{nt};
    if isfield(noiseSplit, fld)
        files = noiseSplit.(fld).(splitKey);
    else
        files = {};
    end
    noiseWins.(fld) = collect_windows(files, FS, WIN, HOP);
    % If no real files: synthesize
    if isempty(noiseWins.(fld)) && ismember(fld, {'tank','engine'})
        noiseWins.(fld) = synth_noise_windows(fld, FS, WIN, 50);
    end
end

speechWins = noiseWins.speech;

% ── DRONE examples ────────────────────────────────────────────────────────
% Collect all drone windows for this split
droneWins = collect_windows(droneFiles, FS, WIN, HOP);
if isempty(droneWins)
    error('create_phase2_v3_dataset: no drone windows available for %s split', splitKey);
end
fprintf('  [%s] Drone base windows: %d\n', splitKey, size(droneWins,2));

% Estimate how many base windows we can use
% Each window → N_VIEWS views × (1 clean + N_noise_types × N_snr augmented)
% To stay within maxPerClass we compute a budget
augPerWindow = N_VIEWS * (1 + numel(noiseTypes) * numel(SNR_LEVELS));
maxBaseWindows = max(1, floor(maxPerClass / augPerWindow));
useWins = min(size(droneWins, 2), maxBaseWindows);
droneWins = droneWins(:, 1:useWins);

for wi = 1:size(droneWins, 2)
    if nDrone >= maxPerClass, break; end
    win = double(droneWins(:, wi));

    % 1. Clean drone (all 5 views)
    for vi = 1:N_VIEWS
        if nDrone >= maxPerClass, break; end
        filt = apply_filter_view(win, vi, FS);
        lm   = extract_logmel_phase2_v3(filt, FS);
        if check_size(lm, SPEC_H, SPEC_W)
            nDrone = nDrone + 1;
            Xdrone(:,:,1,nDrone) = lm;
        end
    end

    % 2. Drone + noise mixtures
    for nt = 1:numel(noiseTypes)
        ntype = noiseTypes{nt};
        nwins = noiseWins.(ntype);
        if isempty(nwins), continue; end

        for si = 1:numel(SNR_LEVELS)
            if nDrone >= maxPerClass, break; end
            snrDb = SNR_LEVELS(si);
            % pick random noise window
            nIdx  = randi(size(nwins, 2));
            nwin  = double(nwins(:, nIdx));
            mixed = mix_at_snr(win, nwin, snrDb);
            mixed = double(mixed(:));

            % pick random view for this augmented example
            vi = randi(N_VIEWS);
            filt = apply_filter_view(mixed, vi, FS);
            lm   = extract_logmel_phase2_v3(filt, FS);
            if check_size(lm, SPEC_H, SPEC_W)
                nDrone = nDrone + 1;
                Xdrone(:,:,1,nDrone) = lm;
            end

            % Also add drone+tank+speech if both exist (hard case)
            if strcmp(ntype,'tank') && ~isempty(speechWins)
                if nDrone >= maxPerClass, break; end
                sIdx   = randi(size(speechWins, 2));
                swin   = double(speechWins(:, sIdx));
                triple = double(mix_at_snr(mixed, swin, snrDb));
                vi2    = randi(N_VIEWS);
                filt2  = apply_filter_view(triple, vi2, FS);
                lm2    = extract_logmel_phase2_v3(filt2, FS);
                if check_size(lm2, SPEC_H, SPEC_W)
                    nDrone = nDrone + 1;
                    Xdrone(:,:,1,nDrone) = lm2;
                end
            end
        end
    end
end

% ── NO-DRONE examples ─────────────────────────────────────────────────────
% 1. Pure noise (all types, all views)
for nt = 1:numel(noiseTypes)
    ntype = noiseTypes{nt};
    nwins = noiseWins.(ntype);
    if isempty(nwins), continue; end
    for wi2 = 1:size(nwins, 2)
        if nNodrone >= maxPerClass, break; end
        nwin = double(nwins(:, wi2));
        for vi = 1:N_VIEWS
            if nNodrone >= maxPerClass, break; end
            filt = apply_filter_view(nwin, vi, FS);
            lm   = extract_logmel_phase2_v3(filt, FS);
            if check_size(lm, SPEC_H, SPEC_W)
                nNodrone = nNodrone + 1;
                Xnodrone(:,:,1,nNodrone) = lm;
            end
        end
    end
end

% 2. Original no_drone clips
ndWins = collect_windows(nodroneFiles, FS, WIN, HOP);
for wi2 = 1:size(ndWins, 2)
    if nNodrone >= maxPerClass, break; end
    nwin = double(ndWins(:, wi2));
    for vi = 1:N_VIEWS
        if nNodrone >= maxPerClass, break; end
        filt = apply_filter_view(nwin, vi, FS);
        lm   = extract_logmel_phase2_v3(filt, FS);
        if check_size(lm, SPEC_H, SPEC_W)
            nNodrone = nNodrone + 1;
            Xnodrone(:,:,1,nNodrone) = lm;
        end
    end
end

% 3. Hard negatives: noise+speech (no drone)
hardPairs = {{'tank','speech'},{'engine','speech'},{'traffic','speech'}};
for hp = 1:numel(hardPairs)
    typeA = hardPairs{hp}{1};
    typeB = hardPairs{hp}{2};
    winsA = noiseWins.(typeA);
    winsB = noiseWins.(typeB);
    if isempty(winsA) || isempty(winsB), continue; end
    nHard = min(size(winsA,2), 20);
    for wi2 = 1:nHard
        if nNodrone >= maxPerClass, break; end
        winA  = double(winsA(:, wi2));
        idxB  = randi(size(winsB,2));
        winB  = double(winsB(:, idxB));
        mixed = double(mix_at_snr(winA, winB, 0));
        vi    = randi(N_VIEWS);
        filt  = apply_filter_view(mixed, vi, FS);
        lm    = extract_logmel_phase2_v3(filt, FS);
        if check_size(lm, SPEC_H, SPEC_W)
            nNodrone = nNodrone + 1;
            Xnodrone(:,:,1,nNodrone) = lm;
        end
    end
end

% Trim to actual count
Xdrone   = Xdrone(:,:,:,1:nDrone);
Xnodrone = Xnodrone(:,:,:,1:nNodrone);
fprintf('  [%s] Drone=%d  No-drone=%d\n', splitKey, nDrone, nNodrone);
end


% =========================================================================
%  assemble: concatenate drone + no_drone, shuffle, cap at maxPerClass each
% =========================================================================
function [X, Y] = assemble(Xdrone, Xnodrone, maxPerClass)
nD  = min(size(Xdrone,4),   maxPerClass);
nND = min(size(Xnodrone,4), maxPerClass);

% Random subsample if over limit
if size(Xdrone,4) > nD
    idx    = randperm(size(Xdrone,4), nD);
    Xdrone = Xdrone(:,:,:,idx);
end
if size(Xnodrone,4) > nND
    idx     = randperm(size(Xnodrone,4), nND);
    Xnodrone = Xnodrone(:,:,:,idx);
end

X = cat(4, Xdrone, Xnodrone);
Y = categorical([repmat({'drone'},    1, size(Xdrone,4)), ...
                 repmat({'no_drone'}, 1, size(Xnodrone,4))]);
% Shuffle
perm = randperm(size(X,4));
X    = X(:,:,:,perm);
Y    = Y(perm);
Y    = Y(:);   % ensure column vector
end


% =========================================================================
%  apply_filter_view: apply one of the 5 spectral views to a signal
% =========================================================================
function y = apply_filter_view(x, viewIdx, fs)
x = double(x(:));
switch viewIdx
    case 1  % raw
        y = x;
    case 2  % highpass 150 Hz
        [b, a] = butter(4, 150/(fs/2), 'high');
        y = filtfilt(b, a, x);
    case 3  % highpass 250 Hz
        [b, a] = butter(4, 250/(fs/2), 'high');
        y = filtfilt(b, a, x);
    case 4  % bandpass 200-6000 Hz
        upperCut = min(6000, 0.45*fs);
        [b, a] = butter(4, [200, upperCut]/(fs/2));
        y = filtfilt(b, a, x);
    case 5  % bandpass 500-6000 Hz
        upperCut = min(6000, 0.45*fs);
        [b, a] = butter(4, [500, upperCut]/(fs/2));
        y = filtfilt(b, a, x);
    otherwise
        y = x;
end
% DC removal + peak normalisation
y = y - mean(y);
pk = max(abs(y));
if pk > 1e-6, y = y / pk; end
end


% =========================================================================
%  collect_windows: read WAV files, window into 1-second chunks
% =========================================================================
function wins = collect_windows(files, fs, winSamples, hopSamples)
wins = [];
for fi = 1:numel(files)
    try
        [audio, sr] = audioread(files{fi});
    catch
        continue
    end
    if size(audio,2) > 1, audio = mean(audio,2); end
    audio = double(audio(:));
    if sr ~= fs, audio = resample(audio, fs, sr); end
    audio = audio - mean(audio);
    pk = max(abs(audio));
    if pk < 1e-5, continue; end
    audio = audio / pk;
    nSamples = length(audio);
    starts   = 1 : hopSamples : (nSamples - winSamples + 1);
    for si = 1:numel(starts)
        s    = starts(si);
        chunk = audio(s : s + winSamples - 1);
        wins  = [wins, single(chunk)]; %#ok<AGROW>
    end
end
end


% =========================================================================
%  synth_noise_windows: generate synthetic noise windows (fallback)
% =========================================================================
function wins = synth_noise_windows(noiseType, fs, winSamples, nWins)
wins = zeros(winSamples, nWins, 'single');
for k = 1:nWins
    t0 = (k-1) * winSamples / fs;
    switch noiseType
        case 'tank'
            wins(:,k) = synth_tank(winSamples, fs, t0);
        case 'engine'
            wins(:,k) = synth_engine(winSamples, fs, t0);
        otherwise
            % Pink-ish noise
            w = randn(winSamples, 1);
            w = w / (max(abs(w)) + eps);
            wins(:,k) = single(w);
    end
end
end

function s = synth_tank(n, fs, t0)
    t   = t0 + (0:n-1)'/fs;
    rpm = 1 + 0.04*sin(2*pi*0.3*t);
    f0  = 45;
    eng = 0.55*sin(2*pi*f0*rpm.*t) + 0.25*sin(2*pi*2*f0*rpm.*t) + ...
          0.12*sin(2*pi*3*f0*rpm.*t) + 0.08*sin(2*pi*4*f0*rpm.*t);
    clank = zeros(n,1);
    stepSz = max(1, round(fs*0.15));
    for pos = 1:stepSz:n
        b = min(round(fs*0.01), n-pos+1);
        clank(pos:pos+b-1) = randn(b,1)*0.4;
    end
    tap = round(fs*0.004);
    lp  = filter(ones(tap,1)/tap, 1, randn(n,1));
    raw = eng + clank + lp*0.3;
    pk  = max(abs(raw));
    s   = single(raw / (pk+eps) * 0.85);
end

function s = synth_engine(n, fs, t0)
    t   = t0 + (0:n-1)'/fs;
    rpm = 1 + 0.03*sin(2*pi*1.5*t);
    f0  = 80;
    eng = 0.5*sin(2*pi*f0*rpm.*t) + 0.3*sin(2*pi*2*f0*rpm.*t) + ...
          0.1*sin(2*pi*3*f0*rpm.*t);
    tap = round(fs*0.002);
    lp  = filter(ones(tap,1)/tap, 1, randn(n,1));
    raw = eng + lp*0.2;
    pk  = max(abs(raw));
    s   = single(raw / (pk+eps) * 0.85);
end


% =========================================================================
%  scan_wavs: return cell array of WAV file paths in a folder
% =========================================================================
function files = scan_wavs(folder)
files = {};
if ~exist(folder, 'dir'), return; end
d = dir(fullfile(folder, '*.wav'));
for fi = 1:numel(d)
    files{end+1} = fullfile(folder, d(fi).name); %#ok<AGROW>
end
end


% =========================================================================
%  split_files: split file list into train/val/test by source file
% =========================================================================
function [tr, va, te] = split_files(files, fracTrain, fracVal)
n    = numel(files);
if n == 0
    tr = {}; va = {}; te = {}; return;
end
perm = randperm(n);
nTr  = max(1, round(n * fracTrain));
nVa  = max(1, round(n * fracVal));
nTe  = max(1, n - nTr - nVa);
tr   = files(perm(1:nTr));
va   = files(perm(nTr+1 : nTr+nVa));
te   = files(perm(nTr+nVa+1 : nTr+nVa+nTe));
end


% =========================================================================
%  check_size: validate feature matrix dimensions
% =========================================================================
function ok = check_size(lm, H, W)
ok = (size(lm,1) == H) && (size(lm,2) == W) && ~any(isnan(lm(:)));
end


% =========================================================================
%  iif: inline if (helper for cache file naming)
% =========================================================================
function out = iif(cond, a, b)
if cond, out = a; else, out = b; end
end
