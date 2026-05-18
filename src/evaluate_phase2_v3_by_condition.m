function condMetrics = evaluate_phase2_v3_by_condition(net, config, rootDir, resultsDir)
% EVALUATE_PHASE2_V3_BY_CONDITION  Per-condition performance report.
%
%   condMetrics = evaluate_phase2_v3_by_condition(net, config, rootDir, resultsDir)
%
%   Generates test examples for each specific condition (drone+tank, tank-alone,
%   etc.) and reports detection rate / false alarm rate at multiple thresholds.
%
%   POSITIVE CONDITIONS
%     drone alone, drone+tank 0dB, drone+tank -5dB, drone+tank -10dB,
%     drone+tank -20dB, drone+engine, drone+speech, drone+tank+speech
%
%   NEGATIVE CONDITIONS
%     tank alone, engine alone, speech alone, crowd alone, wind alone,
%     traffic alone, tank+speech, engine+speech
%
%   SAVED FILES
%     results/phase2_v3/condition_metrics.mat
%     results/phase2_v3/condition_summary_table.csv

if ~exist(resultsDir, 'dir'), mkdir(resultsDir); end

FS          = 16000;
WIN_SAMPLES = 16000;
HOP_SAMPLES = 8000;
N_WINDOWS   = 400;         % windows per condition
N_VIEWS     = 5;
THRESHOLDS  = [0.3, 0.4, 0.5, 0.6, 0.7];
DRONE_DIR   = fullfile(rootDir, 'data', 'raw', 'drone');
NOISE_BASE  = fullfile(rootDir, 'data', 'noise');

fprintf('[CondEval] Loading drone windows ...\n');
droneWins = load_drone_windows(DRONE_DIR, FS, WIN_SAMPLES, HOP_SAMPLES, N_WINDOWS);
if isempty(droneWins)
    error('evaluate_phase2_v3_by_condition: no drone windows available');
end

% Determine feature size from a test prediction
testLM  = extract_logmel_phase2_v3(double(droneWins(:,1)), FS);
SPEC_H  = size(testLM,1);
SPEC_W  = size(testLM,2);

% Determine drone class index
try
    lastLayer = net.Layers(end);
    if isprop(lastLayer,'Classes')
        cls = string(lastLayer.Classes);
        droneIdx = find(strcmpi(cls,'drone'),1);
        if isempty(droneIdx), droneIdx = 1; end
    else
        droneIdx = 1;
    end
catch
    droneIdx = 1;
end

% ── Define conditions ─────────────────────────────────────────────────────
conditions = {
    'drone_alone',           true,  {},         0;
    'drone_tank_0dB',        true,  {'tank'},    0;
    'drone_tank_minus5dB',   true,  {'tank'},   -5;
    'drone_tank_minus10dB',  true,  {'tank'},  -10;
    'drone_tank_minus20dB',  true,  {'tank'},  -20;
    'drone_engine',          true,  {'engine'},  0;
    'drone_speech',          true,  {'speech'},  0;
    'drone_tank_speech',     true,  {'tank','speech'}, 0;
    'tank_alone',            false, {'tank'},    0;
    'engine_alone',          false, {'engine'},  0;
    'speech_alone',          false, {'speech'},  0;
    'crowd_alone',           false, {'crowd'},   0;
    'wind_alone',            false, {'wind'},    0;
    'traffic_alone',         false, {'traffic'}, 0;
    'tank_speech',           false, {'tank','speech'}, 0;
    'engine_speech',         false, {'engine','speech'}, 0;
};
% columns: name, isPositive, noiseTypes, snrDb

condMetrics = struct();
csvRows = {'condition,isPositive,meanProb,maxProb,detect_at_0.5,FA_at_0.5,detect_at_0.4,FA_at_0.4'};

fprintf('[CondEval] Evaluating %d conditions ...\n', size(conditions,1));

for ci = 1:size(conditions,1)
    condName   = conditions{ci,1};
    isPositive = conditions{ci,2};
    noiseTypes = conditions{ci,3};
    snrDb      = conditions{ci,4};

    fprintf('  [%2d/%d] %s ...', ci, size(conditions,1), condName);

    % ── Generate test windows for this condition ──────────────────────────
    audioWins = make_condition_windows(droneWins, noiseTypes, snrDb, ...
                    isPositive, NOISE_BASE, FS, WIN_SAMPLES, N_WINDOWS);
    if isempty(audioWins)
        fprintf(' SKIPPED (no source audio)\n');
        continue;
    end

    % ── Run inference (multiview: all 5 views, take max of filtered views) –
    probs = zeros(1, size(audioWins,2));
    for wi = 1:size(audioWins,2)
        win = double(audioWins(:,wi));
        viewProbs = zeros(1, N_VIEWS);
        for vi = 1:N_VIEWS
            filt = apply_filter_view(win, vi, FS);
            lm   = extract_logmel_phase2_v3(filt, FS);
            if size(lm,1)==SPEC_H && size(lm,2)==SPEC_W
                X = single(reshape(lm, SPEC_H, SPEC_W, 1, 1));
                try
                    sc = predict(net, X, 'MiniBatchSize', 1);
                    viewProbs(vi) = double(sc(droneIdx));
                catch
                    viewProbs(vi) = 0;
                end
            end
        end
        % Weighted combination (same as combine_multiview_scores)
        weights = [0.05, 0.20, 0.25, 0.35, 0.15];
        probs(wi) = sum(weights .* viewProbs);
    end

    % ── Compute metrics at each threshold ─────────────────────────────────
    thMetrics = struct();
    for ti = 1:numel(THRESHOLDS)
        thr  = THRESHOLDS(ti);
        dets = probs > thr;
        fname = sprintf('thr_%02d', round(thr*10));
        if isPositive
            thMetrics.(fname).recall = mean(dets);
            thMetrics.(fname).FA     = NaN;
        else
            thMetrics.(fname).recall = NaN;
            thMetrics.(fname).FA     = mean(dets);
        end
    end

    cm = struct();
    cm.condName   = condName;
    cm.isPositive = isPositive;
    cm.meanProb   = mean(probs);
    cm.maxProb    = max(probs);
    cm.probs      = probs;
    cm.thresholds = THRESHOLDS;
    cm.thMetrics  = thMetrics;

    condMetrics.(condName) = cm;

    % Print one-liner
    det50 = mean(probs > 0.5) * 100;
    if isPositive
        fprintf(' recall@0.5 = %5.1f%%  meanProb=%.3f\n', det50, mean(probs));
    else
        fprintf(' FA@0.5    = %5.1f%%  meanProb=%.3f\n', det50, mean(probs));
    end

    csvRows{end+1} = sprintf('%s,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f', ...
        condName, isPositive, mean(probs), max(probs), ...
        mean(probs>0.5), iif(~isPositive, mean(probs>0.5), NaN), ...
        mean(probs>0.4), iif(~isPositive, mean(probs>0.4), NaN)); %#ok<AGROW>
end

% ── Print summary table ───────────────────────────────────────────────────
fprintf('\n  %-30s  %8s  %8s  %8s\n', 'Condition', 'MeanProb', 'Recall@.5', 'FA@.5');
fprintf('  %s\n', repmat('-', 1, 60));
cnames = fieldnames(condMetrics);
for ci = 1:numel(cnames)
    cm = condMetrics.(cnames{ci});
    det50 = mean(cm.probs > 0.5) * 100;
    if cm.isPositive
        fprintf('  %-30s  %8.4f  %7.1f%%  %8s\n', cnames{ci}, cm.meanProb, det50, '—');
    else
        fprintf('  %-30s  %8.4f  %8s  %7.1f%%\n', cnames{ci}, cm.meanProb, '—', det50);
    end
end
fprintf('\n');

% ── Save ──────────────────────────────────────────────────────────────────
save(fullfile(resultsDir, 'condition_metrics.mat'), 'condMetrics');

csvFile = fullfile(resultsDir, 'condition_summary_table.csv');
fid = fopen(csvFile, 'w');
for ri = 1:numel(csvRows)
    fprintf(fid, '%s\n', csvRows{ri});
end
fclose(fid);

fprintf('[CondEval] Saved:\n  %s\n  %s\n', ...
    fullfile(resultsDir,'condition_metrics.mat'), csvFile);

end   % main function


% =========================================================================
%  make_condition_windows: generate audio windows for a specific condition
% =========================================================================
function wins = make_condition_windows(droneWins, noiseTypes, snrDb, ...
    isPositive, noiseBase, fs, winSamples, nWins)

wins = [];

if isPositive
    % Signal is drone; add noises on top
    src = droneWins;
else
    % Signal is noise only (first noise type)
    if isempty(noiseTypes)
        wins = []; return;
    end
    src = load_noise_windows(noiseTypes{1}, noiseBase, fs, winSamples, nWins);
    if isempty(src)
        src = synth_noise_windows_local(noiseTypes{1}, fs, winSamples, nWins);
    end
    noiseTypes = noiseTypes(2:end);  % remaining types to add on top
end

nSrc = size(src,2);
if nSrc == 0, return; end

% Cap to nWins
if nSrc > nWins, src = src(:, randperm(nSrc, nWins)); nSrc = nWins; end

wins = zeros(winSamples, nSrc, 'single');
for wi = 1:nSrc
    mixed = double(src(:, wi));
    for ni = 1:numel(noiseTypes)
        nwins = load_noise_windows(noiseTypes{ni}, noiseBase, fs, winSamples, 20);
        if isempty(nwins)
            nwins = synth_noise_windows_local(noiseTypes{ni}, fs, winSamples, 20);
        end
        if isempty(nwins), continue; end
        nwin  = double(nwins(:, randi(size(nwins,2))));
        mixed = double(mix_at_snr(mixed, nwin, snrDb));
    end
    wins(:,wi) = single(mixed(:));
end
end


% =========================================================================
%  load_drone_windows: load drone wav files into window matrix
% =========================================================================
function wins = load_drone_windows(droneDir, fs, winSamples, hopSamples, maxWins)
wins = [];
if ~exist(droneDir,'dir'), return; end
d = dir(fullfile(droneDir, '*.wav'));
for fi = 1:numel(d)
    if size(wins,2) >= maxWins, break; end
    try
        [audio, sr] = audioread(fullfile(droneDir, d(fi).name));
    catch, continue; end
    if size(audio,2)>1, audio=mean(audio,2); end
    audio = double(audio(:));
    if sr~=fs, audio=resample(audio,fs,sr); end
    audio = audio - mean(audio);
    pk = max(abs(audio));
    if pk<1e-5, continue; end
    audio = audio / pk;
    starts = 1:hopSamples:(length(audio)-winSamples+1);
    for si = 1:numel(starts)
        if size(wins,2)>=maxWins, break; end
        wins = [wins, single(audio(starts(si):starts(si)+winSamples-1))]; %#ok<AGROW>
    end
end
end


% =========================================================================
%  load_noise_windows: load noise wav files for a given type
% =========================================================================
function wins = load_noise_windows(noiseType, noiseBase, fs, winSamples, maxWins)
wins = [];
folder = fullfile(noiseBase, noiseType);
if ~exist(folder,'dir'), return; end
d = dir(fullfile(folder,'*.wav'));
for fi = 1:numel(d)
    if size(wins,2) >= maxWins, break; end
    try
        [audio,sr] = audioread(fullfile(folder,d(fi).name));
    catch, continue; end
    if size(audio,2)>1, audio=mean(audio,2); end
    audio = double(audio(:));
    if sr~=fs, audio=resample(audio,fs,sr); end
    audio = audio-mean(audio);
    pk=max(abs(audio));
    if pk<1e-5,continue;end
    audio=audio/pk;
    if length(audio)<winSamples
        reps=ceil(winSamples/length(audio));
        audio=repmat(audio,reps,1);
    end
    wins=[wins,single(audio(1:winSamples))]; %#ok<AGROW>
end
end


% =========================================================================
%  synth_noise_windows_local: synthesize noise windows (fallback)
% =========================================================================
function wins = synth_noise_windows_local(noiseType, fs, winSamples, nWins)
wins = zeros(winSamples, nWins, 'single');
for k = 1:nWins
    t0 = (k-1)*winSamples/fs;
    switch noiseType
        case 'tank'
            wins(:,k) = synth_tank_local(winSamples, fs, t0);
        case 'engine'
            wins(:,k) = synth_engine_local(winSamples, fs, t0);
        otherwise
            w = randn(winSamples,1);
            wins(:,k) = single(w/(max(abs(w))+eps));
    end
end
end

function s = synth_tank_local(n,fs,t0)
    t=t0+(0:n-1)'/fs; rpm=1+0.04*sin(2*pi*0.3*t); f0=45;
    eng=0.55*sin(2*pi*f0*rpm.*t)+0.25*sin(2*pi*2*f0*rpm.*t)+...
        0.12*sin(2*pi*3*f0*rpm.*t)+0.08*sin(2*pi*4*f0*rpm.*t);
    clank=zeros(n,1); step=max(1,round(fs*0.15));
    for pos=1:step:n; b=min(round(fs*0.01),n-pos+1); clank(pos:pos+b-1)=randn(b,1)*0.4; end
    tap=max(1,round(fs*0.004)); lp=filter(ones(tap,1)/tap,1,randn(n,1));
    raw=eng+clank+lp*0.3; s=single(raw/(max(abs(raw))+eps)*0.85);
end

function s = synth_engine_local(n,fs,t0)
    t=t0+(0:n-1)'/fs; rpm=1+0.03*sin(2*pi*1.5*t); f0=80;
    eng=0.5*sin(2*pi*f0*rpm.*t)+0.3*sin(2*pi*2*f0*rpm.*t)+0.1*sin(2*pi*3*f0*rpm.*t);
    tap=max(1,round(fs*0.002)); lp=filter(ones(tap,1)/tap,1,randn(n,1));
    raw=eng+lp*0.2; s=single(raw/(max(abs(raw))+eps)*0.85);
end


% =========================================================================
%  apply_filter_view: apply one of the 5 spectral views
% =========================================================================
function y = apply_filter_view(x, viewIdx, fs)
x = double(x(:));
switch viewIdx
    case 2, [b,a]=butter(4,150/(fs/2),'high'); y=filtfilt(b,a,x);
    case 3, [b,a]=butter(4,250/(fs/2),'high'); y=filtfilt(b,a,x);
    case 4, uc=min(6000,0.45*fs); [b,a]=butter(4,[200,uc]/(fs/2)); y=filtfilt(b,a,x);
    case 5, uc=min(6000,0.45*fs); [b,a]=butter(4,[500,uc]/(fs/2)); y=filtfilt(b,a,x);
    otherwise, y=x;
end
y=y-mean(y); pk=max(abs(y)); if pk>1e-6, y=y/pk; end
end

function out=iif(cond,a,b); if cond, out=a; else, out=b; end; end
