function logMel = extract_logmel_phase2_v3(audio, fs)
% EXTRACT_LOGMEL_PHASE2_V3  Compute log-mel spectrogram for Phase 2v3 training.
%
%   logMel = extract_logmel_phase2_v3(audio, fs)
%
%   INPUT
%     audio - mono audio vector (any length, any orientation)
%     fs    - sample rate in Hz (will be resampled to 16000 if needed)
%
%   OUTPUT
%     logMel - [64 × nFrames] single matrix  (log10 mel spectrogram)
%
%   SETTINGS  (must match predict_drone_multiview.m exactly)
%     fsTarget      = 16000 Hz
%     WindowLength  = 400 samples  (25 ms)
%     OverlapLength = 240 samples  (15 ms)   → hop = 160 (10 ms)
%     FFTLength     = 512
%     NumBands      = 64
%     logMel        = log10(S + eps)

% ── Constants ─────────────────────────────────────────────────────────────
FS_TARGET      = 16000;
WIN_LENGTH     = 400;    % round(0.025 * 16000)
OVERLAP_LENGTH = 240;    % round(0.015 * 16000)
FFT_LENGTH     = 512;
NUM_BANDS      = 64;

% ── Ensure mono double column ──────────────────────────────────────────────
if ~isvector(audio), audio = mean(audio, 2); end
audio = double(audio(:));

% ── Resample if needed ─────────────────────────────────────────────────────
if fs ~= FS_TARGET
    audio = resample(audio, FS_TARGET, fs);
end

% ── Compute log-mel spectrogram ────────────────────────────────────────────
try
    % Preferred: Audio Toolbox melSpectrogram
    [S, ~, ~] = melSpectrogram(audio, FS_TARGET, ...
        'Window',        hann(WIN_LENGTH, 'periodic'), ...
        'OverlapLength', OVERLAP_LENGTH, ...
        'FFTLength',     FFT_LENGTH, ...
        'NumBands',      NUM_BANDS, ...
        'FrequencyRange',[0, FS_TARGET/2]);
    logMel = single(log10(S + eps));
catch
    % Fallback: manual mel filterbank (Signal Processing Toolbox)
    logMel = compute_logmel_manual(audio, FS_TARGET, ...
                                   WIN_LENGTH, OVERLAP_LENGTH, ...
                                   FFT_LENGTH, NUM_BANDS);
end

end   % end of main function


% =========================================================================
%  Local fallback: manual mel filterbank
% =========================================================================
function logMel = compute_logmel_manual(x, fs, winLen, overlapLen, fftLen, numBands)
    hopLen  = winLen - overlapLen;
    nFrames = max(0, floor((length(x) - winLen) / hopLen) + 1);
    win     = hann(winLen);
    melFB   = build_mel_filterbank(numBands, fftLen, fs);

    logMel  = zeros(numBands, nFrames, 'single');
    for k = 1:nFrames
        s     = (k-1)*hopLen + 1;
        frame = x(s : s+winLen-1) .* win;
        X     = fft(frame, fftLen);
        ps    = abs(X(1:fftLen/2+1)).^2 / winLen;
        melS  = melFB * ps;
        logMel(:, k) = single(log10(melS + eps));
    end
end

function fb = build_mel_filterbank(numBands, fftLen, fs)
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
                fb(m,k) = (f - lo) / (ctr - lo + eps);
            elseif f > ctr && f <= hi
                fb(m,k) = (hi - f) / (hi - ctr + eps);
            end
        end
    end
end

function m = hz2mel(f), m = 2595 * log10(1 + f/700); end
function f = mel2hz(m), f = 700 * (10.^(m/2595) - 1); end
