function viewMetrics = evaluate_phase2_v3_by_view(net, config, rootDir, resultsDir)
% EVALUATE_PHASE2_V3_BY_VIEW  Per-view performance breakdown.
%
%   viewMetrics = evaluate_phase2_v3_by_view(net, config, rootDir, resultsDir)
%
%   For each of the 5 spectral views, evaluates the model on:
%     drone alone, tank alone, drone+tank 0dB, drone+tank -5dB,
%     drone+tank -10dB, engine alone, speech alone
%
%   PURPOSE: reveals which views contribute most to detection / false alarms.
%   Use this to decide if view-specific thresholds would help.
%
%   SAVED FILES
%     results/phase2_v3/view_metrics.mat
%     results/phase2_v3/view_summary_table.csv

if ~exist(resultsDir, 'dir'), mkdir(resultsDir); end

FS          = 16000;
WIN_SAMPLES = 16000;
HOP_SAMPLES = 8000;
N_WINDOWS   = 300;
THRESHOLD   = 0.5;
NOISE_BASE  = fullfile(rootDir, 'data', 'noise');
DRONE_DIR   = fullfile(rootDir, 'data', 'raw', 'drone');

VIEW_NAMES = {'raw','highpass_150Hz','highpass_250Hz', ...
              'bandpass_200_6000Hz','bandpass_500_6000Hz'};
N_VIEWS    = numel(VIEW_NAMES);

% Determine drone class index
droneIdx = 1;
try
    lastLayer = net.Layers(end);
    if isprop(lastLayer,'Classes')
        cls = string(lastLayer.Classes);
        idx = find(strcmpi(cls,'drone'),1);
        if ~isempty(idx), droneIdx = idx; end
    end
catch; end

% ── Load/generate drone windows ───────────────────────────────────────────
fprintf('[ViewEval] Loading drone windows ...\n');
droneWins = [];
if exist(DRONE_DIR,'dir')
    d = dir(fullfile(DRONE_DIR,'*.wav'));
    for fi = 1:numel(d)
        if size(droneWins,2)>=N_WINDOWS, break; end
        try
            [a,sr]=audioread(fullfile(DRONE_DIR,d(fi).name));
            if size(a,2)>1, a=mean(a,2); end
            a=double(a(:)); if sr~=FS, a=resample(a,FS,sr); end
            a=a-mean(a); pk=max(abs(a)); if pk<1e-5,continue;end
            a=a/pk;
            starts=1:HOP_SAMPLES:(length(a)-WIN_SAMPLES+1);
            for si=1:numel(starts)
                if size(droneWins,2)>=N_WINDOWS,break;end
                droneWins=[droneWins,single(a(starts(si):starts(si)+WIN_SAMPLES-1))]; %#ok<AGROW>
            end
        catch; end
    end
end
if isempty(droneWins)
    error('evaluate_phase2_v3_by_view: no drone audio found in %s', DRONE_DIR);
end
nDW = size(droneWins,2);
fprintf('  %d drone windows loaded\n', nDW);

% ── Synthesize noise windows ──────────────────────────────────────────────
tankWins   = synth_wins('tank',   FS, WIN_SAMPLES, N_WINDOWS);
engineWins = synth_wins('engine', FS, WIN_SAMPLES, N_WINDOWS);
speechWins = load_or_synth('speech', NOISE_BASE, FS, WIN_SAMPLES, N_WINDOWS);

% ── Per-condition windows ─────────────────────────────────────────────────
condMap = {
    'drone_alone',         droneWins,  {},          0;
    'tank_alone',          tankWins,   {},          0;
    'drone_tank_0dB',      droneWins,  tankWins,    0;
    'drone_tank_minus5dB', droneWins,  tankWins,   -5;
    'drone_tank_minus10dB',droneWins,  tankWins,  -10;
    'engine_alone',        engineWins, {},          0;
    'speech_alone',        speechWins, {},          0;
};

% Pre-determine spectrogram size
testLM = extract_logmel_phase2_v3(zeros(WIN_SAMPLES,1), FS);
SPEC_H = size(testLM,1);
SPEC_W = size(testLM,2);

viewMetrics = struct();
csvHeader = 'view,condition,meanProb,detRate';
csvRows   = {csvHeader};

fprintf('[ViewEval] %d views × %d conditions ...\n', N_VIEWS, size(condMap,1));
fprintf('  %-20s', '');
for vi = 1:N_VIEWS
    fprintf('  %-14s', VIEW_NAMES{vi});
end
fprintf('\n  %s\n', repmat('-',1,20+N_VIEWS*16));

for ci = 1:size(condMap,1)
    condName  = condMap{ci,1};
    srcWins   = condMap{ci,2};
    noiseWins = condMap{ci,3};
    snrDb     = condMap{ci,4};
    if isempty(srcWins), continue; end

    nW   = min(size(srcWins,2), N_WINDOWS);
    idx  = randperm(size(srcWins,2), nW);
    src  = srcWins(:, idx);

    % Mix with noise if needed
    if ~isempty(noiseWins)
        mixed = zeros(WIN_SAMPLES, nW, 'single');
        for wi = 1:nW
            ni = randi(size(noiseWins,2));
            mixed(:,wi) = single(mix_at_snr(double(src(:,wi)), double(noiseWins(:,ni)), snrDb));
        end
        src = mixed;
    end

    fprintf('  %-20s', condName);

    % For each view, run inference
    for vi = 1:N_VIEWS
        probs = zeros(1, nW);
        for wi = 1:nW
            win  = double(src(:,wi));
            filt = apply_filter_view(win, vi, FS);
            lm   = extract_logmel_phase2_v3(filt, FS);
            if size(lm,1)==SPEC_H && size(lm,2)==SPEC_W
                X = single(reshape(lm, SPEC_H, SPEC_W, 1, 1));
                try
                    sc = predict(net, X, 'MiniBatchSize',1);
                    probs(wi) = double(sc(droneIdx));
                catch
                    probs(wi) = 0;
                end
            end
        end
        dr = mean(probs > THRESHOLD) * 100;
        mp = mean(probs);
        fprintf('  %5.1f%% (μ=%.3f)', dr, mp);

        % Store
        key = sprintf('%s_view%d', condName, vi);
        viewMetrics.(key).condName   = condName;
        viewMetrics.(key).viewName   = VIEW_NAMES{vi};
        viewMetrics.(key).viewIdx    = vi;
        viewMetrics.(key).meanProb   = mp;
        viewMetrics.(key).detRate    = dr;
        viewMetrics.(key).probs      = probs;
        csvRows{end+1} = sprintf('%s,%s,%.4f,%.4f', VIEW_NAMES{vi}, condName, mp, dr/100); %#ok<AGROW>
    end
    fprintf('\n');
end
fprintf('\n');

% ── Save ──────────────────────────────────────────────────────────────────
save(fullfile(resultsDir,'view_metrics.mat'), 'viewMetrics');

csvFile = fullfile(resultsDir,'view_summary_table.csv');
fid = fopen(csvFile,'w');
for ri = 1:numel(csvRows), fprintf(fid,'%s\n',csvRows{ri}); end
fclose(fid);

fprintf('[ViewEval] Saved:\n  %s\n  %s\n', ...
    fullfile(resultsDir,'view_metrics.mat'), csvFile);
end  % main function


% ── Helpers ──────────────────────────────────────────────────────────────
function wins = synth_wins(type, fs, n, nW)
wins = zeros(n, nW, 'single');
for k=1:nW
    t0=(k-1)*n/fs;
    switch type
        case 'tank'
            t=t0+(0:n-1)'/fs; rpm=1+0.04*sin(2*pi*0.3*t); f0=45;
            eng=0.55*sin(2*pi*f0*rpm.*t)+0.25*sin(2*pi*2*f0*rpm.*t)+...
                0.12*sin(2*pi*3*f0*rpm.*t)+0.08*sin(2*pi*4*f0*rpm.*t);
            clank=zeros(n,1); step=max(1,round(fs*0.15));
            for pos=1:step:n; b=min(round(fs*0.01),n-pos+1); clank(pos:pos+b-1)=randn(b,1)*0.4; end
            tap=max(1,round(fs*0.004)); lp=filter(ones(tap,1)/tap,1,randn(n,1));
            raw=eng+clank+lp*0.3; wins(:,k)=single(raw/(max(abs(raw))+eps)*0.85);
        case 'engine'
            t=t0+(0:n-1)'/fs; rpm=1+0.03*sin(2*pi*1.5*t); f0=80;
            eng=0.5*sin(2*pi*f0*rpm.*t)+0.3*sin(2*pi*2*f0*rpm.*t)+0.1*sin(2*pi*3*f0*rpm.*t);
            tap=max(1,round(fs*0.002)); lp=filter(ones(tap,1)/tap,1,randn(n,1));
            raw=eng+lp*0.2; wins(:,k)=single(raw/(max(abs(raw))+eps)*0.85);
        otherwise
            w=randn(n,1); wins(:,k)=single(w/(max(abs(w))+eps));
    end
end
end

function wins = load_or_synth(type, noiseBase, fs, winSamples, maxW)
wins = [];
folder = fullfile(noiseBase, type);
if exist(folder,'dir')
    d = dir(fullfile(folder,'*.wav'));
    for fi=1:numel(d)
        if size(wins,2)>=maxW,break;end
        try
            [a,sr]=audioread(fullfile(folder,d(fi).name));
            if size(a,2)>1,a=mean(a,2);end
            a=double(a(:)); if sr~=fs,a=resample(a,fs,sr);end
            a=a-mean(a);pk=max(abs(a));if pk<1e-5,continue;end
            a=a/pk;
            if length(a)<winSamples,a=repmat(a,ceil(winSamples/length(a)),1);end
            wins=[wins,single(a(1:winSamples))]; %#ok<AGROW>
        catch;end
    end
end
if isempty(wins)
    wins = synth_wins(type, fs, winSamples, maxW);
end
end

function y = apply_filter_view(x, vi, fs)
x=double(x(:));
switch vi
    case 2,[b,a]=butter(4,150/(fs/2),'high');y=filtfilt(b,a,x);
    case 3,[b,a]=butter(4,250/(fs/2),'high');y=filtfilt(b,a,x);
    case 4,uc=min(6000,0.45*fs);[b,a]=butter(4,[200,uc]/(fs/2));y=filtfilt(b,a,x);
    case 5,uc=min(6000,0.45*fs);[b,a]=butter(4,[500,uc]/(fs/2));y=filtfilt(b,a,x);
    otherwise,y=x;
end
y=y-mean(y);pk=max(abs(y));if pk>1e-6,y=y/pk;end
end
