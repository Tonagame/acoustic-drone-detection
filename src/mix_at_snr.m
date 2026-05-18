function [y, noiseScaled] = mix_at_snr(clean, noise, snrDb)
% MIX_AT_SNR  Mix a clean signal with noise at a specified SNR level.
%
%   [y, noiseScaled] = mix_at_snr(clean, noise, snrDb)
%
%   INPUT
%     clean  - clean signal (mono column vector, pre-resampled to target fs)
%     noise  - noise signal (mono column vector, pre-resampled to target fs)
%     snrDb  - desired signal-to-noise ratio in dB
%              positive = clean louder than noise
%              negative = noise louder than clean
%
%   OUTPUT
%     y          - mixed signal, peak-normalised to ~1
%     noiseScaled - noise scaled to achieve the requested SNR
%
%   SNR DEFINITION
%     SNR = 10 * log10(pClean / pNoise_scaled)
%     => noiseScaled = noise * sqrt(pClean / (pNoise * 10^(snrDb/10)))

% ── Ensure mono double column vectors ─────────────────────────────────────
if ~isvector(clean), clean = mean(clean, 2); end
if ~isvector(noise), noise = mean(noise, 2); end
clean = double(clean(:));
noise = double(noise(:));

% ── Remove DC offset ──────────────────────────────────────────────────────
clean = clean - mean(clean);
noise = noise - mean(noise);

% ── Match lengths ─────────────────────────────────────────────────────────
nC = length(clean);
nN = length(noise);

if nN < nC
    % Repeat noise to cover the clean signal length
    reps  = ceil(nC / nN);
    noise = repmat(noise, reps, 1);
    nN    = length(noise);
end

if nN > nC
    % Random crop noise to match clean length
    maxStart = nN - nC + 1;
    startIdx = randi(maxStart);
    noise = noise(startIdx : startIdx + nC - 1);
end

% ── Compute powers ────────────────────────────────────────────────────────
pClean = mean(clean .^ 2);
pNoise = mean(noise .^ 2);

% ── Scale noise to achieve desired SNR ───────────────────────────────────
% targetNoisePower = pClean / 10^(snrDb/10)
targetNoisePower = pClean / (10 ^ (snrDb / 10));
scaleFactor      = sqrt(targetNoisePower / (pNoise + eps));
noiseScaled      = noise * scaleFactor;

% ── Mix ───────────────────────────────────────────────────────────────────
y = clean + noiseScaled;

% ── Normalize output ──────────────────────────────────────────────────────
pk = max(abs(y));
if pk > 1e-6
    y = y / pk;
end

y           = single(y);
noiseScaled = single(noiseScaled);

end
