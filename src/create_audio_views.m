function [views, viewNames] = create_audio_views(x, fs)
% CREATE_AUDIO_VIEWS  Generate 5 spectral views of one audio window.
%
%   [views, viewNames] = create_audio_views(x, fs)
%
%   INPUT
%     x   - audio vector (mono or stereo, any orientation)
%     fs  - sample rate in Hz
%
%   OUTPUT
%     views     - 1×5 cell array; each cell is a filtered column vector
%     viewNames - 1×5 cell array of descriptive name strings
%
%   THE 5 VIEWS
%     1  raw                 – DC removed, normalised  (full spectrum)
%     2  highpass_150Hz      – removes tank / diesel engine rumble
%     3  highpass_250Hz      – removes more vehicle engine harmonics
%     4  bandpass_200_6000Hz – drone-relevant band (motor + prop)
%     5  bandpass_500_6000Hz – high-frequency propeller harmonics only
%
%   All outputs are normalised column vectors (peak ≈ 1).

% ── Input validation ──────────────────────────────────────────────────────
if nargin < 2
    error('create_audio_views: requires two arguments (x, fs).');
end
if isempty(x)
    error('create_audio_views: input audio is empty.');
end
if fs <= 0
    error('create_audio_views: fs must be positive.');
end

% ── Convert to mono double column vector ──────────────────────────────────
if ~isvector(x)
    % matrix: assume columns are channels
    x = mean(x, 2);
end
x = double(x(:));          % force column, double precision
if size(x, 2) > 1          % still stereo row → average
    x = mean(x, 2);
end

% ── Remove DC offset ─────────────────────────────────────────────────────
x = x - mean(x);

% ── Safe peak normalisation ───────────────────────────────────────────────
pk = max(abs(x));
if pk < 1e-6
    warning('create_audio_views: input is near-silent – returning zero views.');
    views     = repmat({zeros(size(x))}, 1, 5);
    viewNames = {'raw', 'highpass_150Hz', 'highpass_250Hz', ...
                 'bandpass_200_6000Hz', 'bandpass_500_6000Hz'};
    return
end
x = x / pk;

% ── Adjust upper BPF cutoff if fs is too low ─────────────────────────────
upperCutoff = 6000;
if upperCutoff >= 0.45 * fs
    upperCutoff = floor(0.45 * fs);
    fprintf('[create_audio_views] WARNING: fs = %d Hz is low. '  , fs);
    fprintf('Upper BPF cutoff adjusted to %d Hz.\n', upperCutoff);
end

% ── Local helper: post-filter normalisation ───────────────────────────────
    function y = norm_col(raw)
        y  = double(raw(:));
        y  = y - mean(y);
        pk2 = max(abs(y));
        if pk2 > 1e-6
            y = y / pk2;
        end
    end

% ── Build the 5 views ─────────────────────────────────────────────────────
viewNames = {'raw', ...
             'highpass_150Hz', ...
             'highpass_250Hz', ...
             'bandpass_200_6000Hz', ...
             'bandpass_500_6000Hz'};
views = cell(1, 5);

% 1 – raw (already clean)
views{1} = x;

% 2 – high-pass at 150 Hz
try
    views{2} = norm_col(highpass(x, 150, fs));
catch ME
    warning('create_audio_views: highpass(150 Hz) failed: %s', ME.message);
    views{2} = x;
end

% 3 – high-pass at 250 Hz
try
    views{3} = norm_col(highpass(x, 250, fs));
catch ME
    warning('create_audio_views: highpass(250 Hz) failed: %s', ME.message);
    views{3} = x;
end

% 4 – band-pass 200 – upperCutoff Hz
try
    views{4} = norm_col(bandpass(x, [200, upperCutoff], fs));
catch ME
    warning('create_audio_views: bandpass([200 %d]) failed: %s', upperCutoff, ME.message);
    % Fallback: high-pass only
    try
        views{4} = norm_col(highpass(x, 200, fs));
    catch
        views{4} = x;
    end
end

% 5 – band-pass 500 – upperCutoff Hz
try
    views{5} = norm_col(bandpass(x, [500, upperCutoff], fs));
catch ME
    warning('create_audio_views: bandpass([500 %d]) failed: %s', upperCutoff, ME.message);
    try
        views{5} = norm_col(highpass(x, 500, fs));
    catch
        views{5} = x;
    end
end

end
