function features = extract_logmel(data)
% EXTRACT_LOGMEL  Compute log-Mel spectrograms for a set of audio windows.
%
%   features = extract_logmel(data)
%
%   Input:
%     data.windows  - cell array of 16000 x 1 audio windows
%     data.labels   - N x 1 categorical array of class labels
%
%   Output:
%     features.X      - single 4-D array  [numBands x numFrames x 1 x N]
%     features.labels - N x 1 categorical array (aligned with 4th dimension)
%
%   Requires: Audio Toolbox (melSpectrogram)

FS          = 16000;
WIN_LEN     = round(0.025 * FS);   % 400 samples  (~25 ms)
OVERLAP_LEN = round(0.015 * FS);   % 240 samples  (hop = 160, ~10 ms)
FFT_LEN     = 512;
NUM_BANDS   = 64;

N = numel(data.windows);
if N == 0
    error('extract_logmel: received an empty windows cell array.');
end

% ------------------------------------------------------------------
% Pre-compute expected output size from one reference window so that
% every feature slice is guaranteed to be the same shape.
% ------------------------------------------------------------------
ref   = double(data.windows{1});
S_ref = melSpectrogram(ref, FS, ...
    'WindowLength',  WIN_LEN, ...
    'OverlapLength', OVERLAP_LEN, ...
    'FFTLength',     FFT_LEN, ...
    'NumBands',      NUM_BANDS);
[H, W] = size(S_ref);
fprintf('Log-Mel shape per window: %d bands x %d frames\n', H, W);

% Pre-allocate
X       = zeros(H, W, 1, N, 'single');
validMask = true(N, 1);

for i = 1 : N
    audio = double(data.windows{i});
    audio = audio(:);   % ensure column vector

    S    = melSpectrogram(audio, FS, ...
        'WindowLength',  WIN_LEN, ...
        'OverlapLength', OVERLAP_LEN, ...
        'FFTLength',     FFT_LEN, ...
        'NumBands',      NUM_BANDS);

    logS = log10(S + eps);

    if ~isequal(size(logS), [H, W])
        % Should not happen for fixed-length 1-second windows;
        % guard against edge cases in malformed files.
        warning('Window %d has unexpected shape [%d x %d] (expected [%d x %d]) – skipped.', ...
            i, size(logS,1), size(logS,2), H, W);
        validMask(i) = false;
        continue;
    end

    X(:, :, 1, i) = single(logS);
end

% Remove any skipped windows
nSkipped = sum(~validMask);
if nSkipped > 0
    warning('extract_logmel: %d window(s) skipped due to size mismatch.', nSkipped);
    X = X(:, :, :, validMask);
end

features.X      = X;
features.labels = data.labels(validMask);
end
