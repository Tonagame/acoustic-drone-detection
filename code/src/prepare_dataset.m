function [trainData, valData, testData] = prepare_dataset(dataRoot)
% PREPARE_DATASET  Load and preprocess WAV files from the DADS dataset.
%
%   [trainData, valData, testData] = prepare_dataset(dataRoot)
%
%   dataRoot must contain two subdirectories:
%     <dataRoot>/drone/     - WAV files labelled "drone"
%     <dataRoot>/no_drone/  - WAV files labelled "no_drone"
%
%   Each output struct has fields:
%     .windows  - cell array of column-vector audio windows (16000 x 1)
%     .labels   - categorical array matching .windows

FS_TARGET   = 16000;
WIN_SAMPLES = FS_TARGET;        % 1-second window
HOP_SAMPLES = WIN_SAMPLES / 2;  % 50% overlap

dronePath   = fullfile(dataRoot, 'drone');
noDronePath = fullfile(dataRoot, 'no_drone');

checkFolder(dronePath,   'drone');
checkFolder(noDronePath, 'no_drone');

% Build audioDatastore - labels come from folder names
ads = audioDatastore({dronePath, noDronePath}, ...
    'LabelSource',        'foldernames', ...
    'IncludeSubfolders',  false);

numFiles = numel(ads.Files);
if numFiles == 0
    error('No WAV files found under %s', dataRoot);
end
fprintf('Found %d audio files (%s).\n', numFiles, dataRoot);

% -------------------------------------------------------------------
% Split by FILE index to avoid data leakage across windows
% -------------------------------------------------------------------
rng(42);   % reproducibility
idx    = randperm(numFiles);
nTrain = round(0.70 * numFiles);
nVal   = round(0.15 * numFiles);
% test = whatever remains

trainIdx = idx(1           : nTrain);
valIdx   = idx(nTrain+1    : nTrain+nVal);
testIdx  = idx(nTrain+nVal+1 : end);

fprintf('File split  ->  train: %d | val: %d | test: %d\n', ...
    numel(trainIdx), numel(valIdx), numel(testIdx));

trainData = processFiles(ads, trainIdx, FS_TARGET, WIN_SAMPLES, HOP_SAMPLES, 'train');
valData   = processFiles(ads, valIdx,   FS_TARGET, WIN_SAMPLES, HOP_SAMPLES, 'val');
testData  = processFiles(ads, testIdx,  FS_TARGET, WIN_SAMPLES, HOP_SAMPLES, 'test');

fprintf('Windows     ->  train: %d | val: %d | test: %d\n', ...
    numel(trainData.windows), numel(valData.windows), numel(testData.windows));
end


% =========================================================================
function data = processFiles(ads, fileIndices, fsTarget, winSamples, hopSamples, splitName)
% Read, preprocess, and slice every file at the given indices into windows.

windows = {};
labels  = {};

for k = 1 : numel(fileIndices)
    i        = fileIndices(k);
    filePath = ads.Files{i};
    label    = string(ads.Labels(i));

    try
        [audio, fs] = audioread(filePath);
    catch ME
        warning('Could not read %s: %s – skipping.', filePath, ME.message);
        continue;
    end

    % Convert to mono
    if size(audio, 2) > 1
        audio = mean(audio, 2);
    end
    audio = audio(:);  % column vector

    % Resample if needed
    if fs ~= fsTarget
        audio = resample(audio, fsTarget, fs);
    end

    % Safe amplitude normalisation (avoid division by zero on silent files)
    peak = max(abs(audio));
    if peak > 0
        audio = audio / peak;
    end

    % Slice into overlapping windows
    nSamples = numel(audio);
    starts   = 1 : hopSamples : nSamples - winSamples + 1;

    for s = starts
        win = audio(s : s + winSamples - 1);   % column, 16000 x 1
        if numel(win) == winSamples
            windows{end+1} = win;    %#ok<AGROW>
            labels{end+1}  = label;  %#ok<AGROW>
        end
    end
end

if isempty(windows)
    error('No valid windows produced for the %s split. Check the audio files.', splitName);
end

data.windows = windows;
data.labels  = categorical(labels(:));   % N x 1 categorical
end


% =========================================================================
function checkFolder(folderPath, name)
% Error if the folder is missing or contains no WAV files.
if ~isfolder(folderPath)
    error([...
        'Required folder not found:\n  %s\n\n' ...
        'Please download the DADS dataset and place WAV files in:\n' ...
        '  data/raw/drone/\n' ...
        '  data/raw/no_drone/\n\n' ...
        'See README_phase1.md for download instructions.'], folderPath);
end

wavFiles = dir(fullfile(folderPath, '*.wav'));
if isempty(wavFiles)
    error([...
        'No WAV files found in the "%s" folder:\n  %s\n\n' ...
        'Expected .wav files directly inside that folder.\n' ...
        'See README_phase1.md for the required folder structure.'], ...
        name, folderPath);
end
end
