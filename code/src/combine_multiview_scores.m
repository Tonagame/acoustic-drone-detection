function [weightedScore, filteredMax, voteCount, detected, debug] = ...
                                            combine_multiview_scores(probs, viewNames)
% COMBINE_MULTIVIEW_SCORES  Combine per-view drone probabilities into one decision.
%
%   [weightedScore, filteredMax, voteCount, detected, debug] = ...
%       combine_multiview_scores(probs, viewNames)
%
%   INPUT
%     probs     - 1×5 vector of drone probabilities (one per spectral view)
%                 Order must match create_audio_views output:
%                   1  raw
%                   2  highpass_150Hz
%                   3  highpass_250Hz
%                   4  bandpass_200_6000Hz
%                   5  bandpass_500_6000Hz
%     viewNames - (optional) 1×5 cell of view name strings (for debug output)
%
%   OUTPUT
%     weightedScore - scalar in [0,1]; weighted combination of all 5 views
%     filteredMax   - scalar in [0,1]; max probability from views 2–5 (no raw)
%     voteCount     - integer in [0,5]; views where prob > VOTE_THRESHOLD (0.60)
%     detected      - logical; true when any detection path triggers
%     debug         - struct with detection reason and best-view info
%
%   DETECTION RULE  (three independent paths – SENSITIVE)
%     detected = true  if  filteredMax   > 0.75          (path A)
%                      OR  weightedScore > 0.60          (path B)
%                      OR  voteCount    >= 2             (path C)
%
%   WEIGHTS (must sum to 1.0)
%     raw                 : 0.05  – full spectrum (very noise-sensitive, low weight)
%     highpass_150Hz      : 0.20  – removes low rumble
%     highpass_250Hz      : 0.25  – removes most engine harmonics
%     bandpass_200_6000Hz : 0.35  – HIGHEST weight: drone core band
%     bandpass_500_6000Hz : 0.15  – high-harmonic focus

% ── Validate input ────────────────────────────────────────────────────────
if nargin < 1 || isempty(probs)
    error('combine_multiview_scores: probs vector is required.');
end
probs = double(probs(:)');          % ensure 1×N row
if numel(probs) ~= 5
    error('combine_multiview_scores: expected 5 probabilities, got %d.', numel(probs));
end
% Clamp to valid probability range
probs = max(0, min(1, probs));

% Default view names if not supplied
if nargin < 2 || isempty(viewNames)
    viewNames = {'raw', 'highpass_150Hz', 'highpass_250Hz', ...
                 'bandpass_200_6000Hz', 'bandpass_500_6000Hz'};
end

% ── Per-view weights ──────────────────────────────────────────────────────
weights = [0.05, 0.20, 0.25, 0.35, 0.15];   % must sum to 1.0

% ── Weighted score (all 5 views) ──────────────────────────────────────────
weightedScore = sum(weights .* probs);

% ── Filtered max: best score from filtered views only (exclude raw) ───────
filteredMax = max(probs(2:5));

% ── Vote count (views confidently declaring "drone") ─────────────────────
VOTE_THRESHOLD = 0.60;  % raised from 0.55 -- engine harmonics in BPF view
                        % no longer accumulate 2 votes at 0.55
voteCount = sum(probs > VOTE_THRESHOLD);

% ── Detection decision (sensitive – three independent paths) ──────────────
FILTERED_THRESHOLD = 0.75;
SCORE_THRESHOLD    = 0.60;
VOTES_NEEDED       = 2;

pathA = filteredMax   > FILTERED_THRESHOLD;
pathB = weightedScore > SCORE_THRESHOLD;
pathC = voteCount    >= VOTES_NEEDED;

detected = pathA || pathB || pathC;

% ── Debug struct ──────────────────────────────────────────────────────────
[bestViewProb, bestViewIdx] = max(probs);
debug.bestViewIndex = bestViewIdx;
debug.bestViewName  = viewNames{bestViewIdx};
debug.bestViewProb  = bestViewProb;
debug.filteredMax   = filteredMax;
debug.weightedScore = weightedScore;
debug.voteCount     = voteCount;

if ~detected
    debug.reason = 'notDetected';
elseif pathA
    debug.reason = 'filteredMax';
elseif pathB
    debug.reason = 'weightedScore';
else
    debug.reason = 'voteCount';
end

end
