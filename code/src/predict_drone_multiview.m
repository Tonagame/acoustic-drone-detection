function result = predict_drone_multiview(x, fs, net)
% PREDICT_DRONE_MULTIVIEW  Run drone CNN on 5 spectral views; combine scores.
%
%   result = predict_drone_multiview(x, fs, net)
%
%   INPUT
%     x   - mono audio window, ideally 1 second (any fs)
%     fs  - sample rate of x (Hz)
%     net - trained MATLAB deep-learning network (SeriesNetwork / DAGNetwork)
%           Class order must have 'drone' as the FIRST class.
%
%   OUTPUT  (struct)
%     result.viewNames     – 1×5 cell of view name strings
%     result.probs         – 1×5 vector of drone probabilities per view
%     result.weightedScore – scalar weighted combination (all 5 views)
%     result.filteredMax   – scalar max probability from filtered views (2–5)
%     result.voteCount     – number of views where prob > 0.55
%     result.detected      – logical detection flag
%     result.debug         – struct: bestViewIndex, bestViewName,
%                                    bestViewProb, reason
%
%   LOG-MEL SETTINGS  (match training pipeline exactly)
%     fsTarget      = 16000 Hz
%     WindowLength  = 400  samples  (25 ms)
%     OverlapLength = 240  samples  (15 ms)  → HopLength = 160 (10 ms)
%     FFTLength     = 512
%     NumBands      = 64
%     logMel        = log10(S + eps)

% ── Constants (must match training) ──────────────────────────────────────
FS_TARGET      = 16000;
WIN_LENGTH     = round(0.025 * FS_TARGET);    % 400 samples
OVERLAP_LENGTH = round(0.015 * FS_TARGET);    % 240 samples
FFT_LENGTH     = 512;
NUM_BANDS      = 64;

% ── Convert to mono double column ─────────────────────────────────────────
if ~isvector(x)
    x = mean(x, 2);
end
x = double(x(:));

% ── Resample to 16 kHz if needed ─────────────────────────────────────────
if fs ~= FS_TARGET
    x = resample(x, FS_TARGET, fs);
end

% ── Create 5 spectral views ───────────────────────────────────────────────
[views, viewNames] = create_audio_views(x, FS_TARGET);

% ── Detect drone class index in the network ───────────────────────────────
droneClassIdx = 1;   % default: assume first class = drone
try
    lastLayer = net.Layers(end);
    if isprop(lastLayer, 'Classes')
        classNames = string(lastLayer.Classes);
        idx = find(strcmpi(classNames, 'drone'), 1);
        if ~isempty(idx)
            droneClassIdx = idx;
        end
    end
catch
    % Cannot inspect classes; keep default index 1
end

% ── Run the CNN on each view ──────────────────────────────────────────────
probs = zeros(1, numel(views));

for v = 1:numel(views)
    audio_v = views{v};

    % Compute Log-Mel spectrogram [NUM_BANDS × nFrames]
    logMel = compute_logmel(audio_v, FS_TARGET, ...
                            WIN_LENGTH, OVERLAP_LENGTH, ...
                            FFT_LENGTH, NUM_BANDS);

    % Format as network input: [Height × Width × Channels × Batch]
    netInput = single(logMel);
    netInput = reshape(netInput, size(netInput,1), size(netInput,2), 1, 1);

    % Run network and extract drone probability
    try
        scores   = predict(net, netInput);   % [1 × nClasses]
        probs(v) = double(scores(droneClassIdx));
    catch ME
        warning('predict_drone_multiview: network failed on view %d (%s): %s', ...
                v, viewNames{v}, ME.message);
        probs(v) = 0;
    end
end

% ── Combine scores ────────────────────────────────────────────────────────
[weightedScore, filteredMax, voteCount, detected, debug] = ...
    combine_multiview_scores(probs, viewNames);

% ── Pack result ───────────────────────────────────────────────────────────
result.viewNames     = viewNames;
result.probs         = probs;
result.weightedScore = weightedScore;
result.filteredMax   = filteredMax;
result.voteCount     = voteCount;
result.detected      = detected;
result.debug         = debug;

end


% =========================================================================
%  Local helper: Log-Mel spectrogram
% =========================================================================
function logMel = compute_logmel(x, fs, winLen, overlapLen, fftLen, numBands)
% COMPUTE_LOGMEL  Return log10 mel spectrogram [numBands × nFrames].
%
%   Matches training:  logMel = log10(S + eps)
%   Uses MATLAB Audio Toolbox melSpectrogram when available,
%   falls back to manual FFT + mel filterbank otherwise.

    try
        % ── Preferred: Audio Toolbox melSpectrogram ───────────────────────
        [S, ~, ~] = melSpectrogram(x, fs, ...
            'Window',        hann(winLen, 'periodic'), ...
            'OverlapLength', overlapLen,               ...
            'FFTLength',     fftLen,                   ...
            'NumBands',      numBands,                 ...
            'FrequencyRange',[0, fs/2]);
        logMel = log10(S + eps);

    catch
        % ── Fallback: manual mel filterbank ──────────────────────────────
        logMel = compute_logmel_manual(x, fs, winLen, overlapLen, fftLen, numBands);
    end
end


function logMel = compute_logmel_manual(x, fs, winLen, overlapLen, fftLen, numBands)
% Manual mel spectrogram without Audio Toolbox.
    hopLen  = winLen - overlapLen;
    nFrames = max(0, floor((length(x) - winLen) / hopLen) + 1);
    win     = hann(winLen);

    % Build mel filterbank
    melFB   = build_mel_filterbank(numBands, fftLen, fs);  % [numBands × (fftLen/2+1)]

    logMel  = zeros(numBands, nFrames, 'single');
    for k = 1:nFrames
        s     = (k-1)*hopLen + 1;
        frame = x(s : s+winLen-1) .* win;
        ps    = (abs(fft(frame, fftLen)(1:fftLen/2+1)).^2) ./ winLen;
        melS  = melFB * ps;
        logMel(:, k) = single(log10(melS + eps));
    end
end


function fb = build_mel_filterbank(numBands, fftLen, fs)
% Build triangular mel filterbank matrix [numBands × (fftLen/2+1)].
    nFreqs   = fftLen/2 + 1;
    freqs    = linspace(0, fs/2, nFreqs);

    melMin   = hz2mel(0);
    melMax   = hz2mel(fs/2);
    melEdges = linspace(melMin, melMax, numBands + 2);
    hzEdges  = mel2hz(melEdges);

    fb = zeros(numBands, nFreqs);
    for m = 1:numBands
        lo  = hzEdges(m);
        ctr = hzEdges(m+1);
        hi  = hzEdges(m+2);
        for k = 1:nFreqs
            f = freqs(k);
            if f >= lo && f <= ctr
                fb(m, k) = (f - lo) / (ctr - lo);
            elseif f > ctr && f <= hi
                fb(m, k) = (hi - f) / (hi - ctr);
            end
        end
    end
end

function m = hz2mel(f)
    m = 2595 * log10(1 + f/700);
end

function f = mel2hz(m)
    f = 700 * (10.^(m/2595) - 1);
end
