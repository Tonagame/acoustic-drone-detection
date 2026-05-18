function compTable = compare_phase2v2_phase2v3(netV3, config, rootDir, resultsDir)
% COMPARE_PHASE2V2_PHASE2V3  Compare Phase 2v2, Phase 2v3 (and Phase 3 if found).
%
%   compTable = compare_phase2v2_phase2v3(netV3, config, rootDir, resultsDir)
%
%   Loads available models and evaluates each on the same 8 canonical conditions.
%   Reports mean drone probability and detection rate at threshold 0.5.
%
%   SAVED FILES
%     results/phase2_v3/model_comparison.mat
%     results/phase2_v3/model_comparison_table.csv

if ~exist(resultsDir,'dir'), mkdir(resultsDir); end

FS          = 16000;
WIN_SAMPLES = 16000;
HOP_SAMPLES = 8000;
N_WINDOWS   = 300;
THRESHOLD   = 0.5;
WEIGHTS     = [0.05, 0.20, 0.25, 0.35, 0.15];
NOISE_BASE  = fullfile(rootDir, 'data', 'noise');
DRONE_DIR   = fullfile(rootDir, 'data', 'raw', 'drone');
MODELS_DIR  = fullfile(rootDir, 'models');

% ── Conditions to compare ─────────────────────────────────────────────────
% {name, isPositive, noiseType, snrDb}
CONDITIONS = {
    'drone_alone',         true,  '',       0;
    'drone_tank_0dB',      true,  'tank',   0;
    'drone_tank_minus5dB', true,  'tank',  -5;
    'drone_tank_minus10dB',true,  'tank', -10;
    'tank_alone',          false, 'tank',   0;
    'engine_alone',        false, 'engine', 0;
    'speech_alone',        false, 'speech', 0;
    'crowd_alone',         false, 'crowd',  0;
};

% ── Load models to compare ────────────────────────────────────────────────
models = struct();
models(1).name = 'phase2v3 (new)';
models(1).net  = netV3;

MODEL_FILES = {
    'drone_cnn_phase2_v2_noise_speech_robust.mat', 'phase2v2';
    'drone_cnn_phase1.mat',                        'phase1';
};
nModels = 1;
for mi = 1:size(MODEL_FILES,1)
    mpath = fullfile(MODELS_DIR, MODEL_FILES{mi,1});
    if ~isfile(mpath), continue; end
    try
        tmp    = load(mpath);
        fnames = fieldnames(tmp);
        loaded = [];
        for fi = 1:numel(fnames)
            cand = tmp.(fnames{fi});
            if isa(cand,'SeriesNetwork')||isa(cand,'DAGNetwork')||isa(cand,'dlnetwork')
                loaded = cand; break;
            end
        end
        if ~isempty(loaded)
            nModels = nModels + 1;
            models(nModels).name = MODEL_FILES{mi,2};
            models(nModels).net  = loaded;
            fprintf('[Compare] Loaded %s\n', MODEL_FILES{mi,2});
        end
    catch ME
        fprintf('[Compare] Could not load %s: %s\n', MODEL_FILES{mi,1}, ME.message);
    end
end

% ── Load / generate audio windows ────────────────────────────────────────
fprintf('[Compare] Preparing audio windows ...\n');
droneWins  = load_drone_wins(DRONE_DIR, FS, WIN_SAMPLES, HOP_SAMPLES, N_WINDOWS);
tankWins   = synth_wins_local('tank',   FS, WIN_SAMPLES, N_WINDOWS);
engineWins = synth_wins_local('engine', FS, WIN_SAMPLES, N_WINDOWS);
speechWins = load_or_synth_local('speech', NOISE_BASE, FS, WIN_SAMPLES, N_WINDOWS);
crowdWins  = load_or_synth_local('crowd',  NOISE_BASE, FS, WIN_SAMPLES, N_WINDOWS);

noiseMap = struct('tank',tankWins,'engine',engineWins,'speech',speechWins,'crowd',crowdWins);

if isempty(droneWins)
    error('compare_phase2v2_phase2v3: no drone windows found in %s', DRONE_DIR);
end

% Pre-compute spectrogram size
testLM = extract_logmel_phase2_v3(zeros(WIN_SAMPLES,1), FS);
SPEC_H = size(testLM,1);
SPEC_W = size(testLM,2);

% ── Evaluate each model on each condition ─────────────────────────────────
% results(model_idx, cond_idx) = struct(meanProb, detRate)
nConds = size(CONDITIONS,1);
compTable = struct();
csvRows   = {'condition,isPositive,' + strjoin(arrayfun(@(m) m.name, models, 'uni', false), ',detRate_', 'uniform', false)};

% Build CSV header
hdrParts = {'condition','isPositive'};
for mi = 1:nModels
    hdrParts{end+1} = [strrep(models(mi).name,' ','_') '_meanProb'];
    hdrParts{end+1} = [strrep(models(mi).name,' ','_') '_detRate'];
end
csvRows = {strjoin(hdrParts, ',')};

fprintf('[Compare] Evaluating %d models × %d conditions ...\n', nModels, nConds);
fprintf('  %-26s', '');
for mi = 1:nModels, fprintf('  %-22s', models(mi).name); end
fprintf('\n  %s\n', repmat('-',1,26+nModels*24));

for ci = 1:nConds
    condName   = CONDITIONS{ci,1};
    isPositive = CONDITIONS{ci,2};
    noiseType  = CONDITIONS{ci,3};
    snrDb      = CONDITIONS{ci,4};

    % Build audio windows for this condition
    if isPositive
        src = droneWins;
        if ~isempty(noiseType) && isfield(noiseMap, noiseType) && ~isempty(noiseMap.(noiseType))
            mixed = zeros(WIN_SAMPLES, min(size(src,2),N_WINDOWS), 'single');
            nW = size(mixed,2);
            nwins = noiseMap.(noiseType);
            for wi=1:nW
                ni=randi(size(nwins,2));
                mixed(:,wi)=single(mix_at_snr(double(src(:,wi)),double(nwins(:,ni)),snrDb));
            end
            src = mixed;
        end
    else
        if isempty(noiseType)||~isfield(noiseMap,noiseType), continue; end
        src = noiseMap.(noiseType);
    end
    if isempty(src), continue; end
    nW = min(size(src,2), N_WINDOWS);
    src = src(:, randperm(size(src,2), nW));

    fprintf('  %-26s', condName);

    rowParts = {condName, num2str(isPositive)};

    for mi = 1:nModels
        mnet     = models(mi).net;
        droneIdx = get_drone_idx(mnet);
        probs    = compute_multiview_probs(mnet, src, droneIdx, WEIGHTS, SPEC_H, SPEC_W, FS);

        mp = mean(probs);
        dr = mean(probs > THRESHOLD) * 100;
        fprintf('  μ=%.3f det=%5.1f%%', mp, dr);

        key = sprintf('%s_%s', strrep(models(mi).name,' ','_'), condName);
        compTable.(key).meanProb = mp;
        compTable.(key).detRate  = dr;
        rowParts{end+1} = sprintf('%.4f', mp);
        rowParts{end+1} = sprintf('%.4f', dr/100);
    end
    fprintf('\n');
    csvRows{end+1} = strjoin(rowParts, ','); %#ok<AGROW>
end
fprintf('\n');

% ── Save ──────────────────────────────────────────────────────────────────
save(fullfile(resultsDir,'model_comparison.mat'), 'compTable');
csvFile = fullfile(resultsDir,'model_comparison_table.csv');
fid = fopen(csvFile,'w');
for ri=1:numel(csvRows), fprintf(fid,'%s\n',csvRows{ri}); end
fclose(fid);
fprintf('[Compare] Saved:\n  %s\n  %s\n', ...
    fullfile(resultsDir,'model_comparison.mat'), csvFile);
end  % main function


% =========================================================================
%  compute_multiview_probs: run all 5 views and return weighted score
% =========================================================================
function probs = compute_multiview_probs(net, srcWins, droneIdx, weights, H, W, fs)
nW    = size(srcWins, 2);
probs = zeros(1, nW);
for wi = 1:nW
    win = double(srcWins(:,wi));
    vp  = zeros(1,5);
    for vi = 1:5
        filt = apply_fv(win, vi, fs);
        lm   = extract_logmel_phase2_v3(filt, fs);
        if size(lm,1)==H && size(lm,2)==W
            X = single(reshape(lm,H,W,1,1));
            try
                sc=predict(net,X,'MiniBatchSize',1); vp(vi)=double(sc(droneIdx));
            catch; end
        end
    end
    probs(wi) = sum(weights.*vp);
end
end

function idx = get_drone_idx(net)
idx = 1;
try
    ll = net.Layers(end);
    if isprop(ll,'Classes')
        cls = string(ll.Classes);
        i = find(strcmpi(cls,'drone'),1);
        if ~isempty(i), idx=i; end
    end
catch; end
end

function y = apply_fv(x,vi,fs)
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

function wins = load_drone_wins(d,fs,win,hop,maxW)
wins=[];
if ~exist(d,'dir'),return;end
files=dir(fullfile(d,'*.wav'));
for fi=1:numel(files)
    if size(wins,2)>=maxW,break;end
    try
        [a,sr]=audioread(fullfile(d,files(fi).name));
        if size(a,2)>1,a=mean(a,2);end
        a=double(a(:));if sr~=fs,a=resample(a,fs,sr);end
        a=a-mean(a);pk=max(abs(a));if pk<1e-5,continue;end
        a=a/pk;
        for s=1:hop:(length(a)-win+1)
            if size(wins,2)>=maxW,break;end
            wins=[wins,single(a(s:s+win-1))]; %#ok<AGROW>
        end
    catch;end
end
end

function wins = synth_wins_local(type,fs,n,nW)
wins=zeros(n,nW,'single');
for k=1:nW
    t0=(k-1)*n/fs;
    switch type
        case 'tank'
            t=t0+(0:n-1)'/fs;rpm=1+0.04*sin(2*pi*0.3*t);f0=45;
            eng=0.55*sin(2*pi*f0*rpm.*t)+0.25*sin(2*pi*2*f0*rpm.*t)+...
                0.12*sin(2*pi*3*f0*rpm.*t)+0.08*sin(2*pi*4*f0*rpm.*t);
            clank=zeros(n,1);step=max(1,round(fs*0.15));
            for pos=1:step:n;b=min(round(fs*0.01),n-pos+1);clank(pos:pos+b-1)=randn(b,1)*0.4;end
            tap=max(1,round(fs*0.004));lp=filter(ones(tap,1)/tap,1,randn(n,1));
            raw=eng+clank+lp*0.3;wins(:,k)=single(raw/(max(abs(raw))+eps)*0.85);
        case 'engine'
            t=t0+(0:n-1)'/fs;rpm=1+0.03*sin(2*pi*1.5*t);f0=80;
            eng=0.5*sin(2*pi*f0*rpm.*t)+0.3*sin(2*pi*2*f0*rpm.*t)+0.1*sin(2*pi*3*f0*rpm.*t);
            tap=max(1,round(fs*0.002));lp=filter(ones(tap,1)/tap,1,randn(n,1));
            raw=eng+lp*0.2;wins(:,k)=single(raw/(max(abs(raw))+eps)*0.85);
        otherwise
            w=randn(n,1);wins(:,k)=single(w/(max(abs(w))+eps));
    end
end
end

function wins = load_or_synth_local(type,noiseBase,fs,winSamples,maxW)
wins=[];
folder=fullfile(noiseBase,type);
if exist(folder,'dir')
    d=dir(fullfile(folder,'*.wav'));
    for fi=1:numel(d)
        if size(wins,2)>=maxW,break;end
        try
            [a,sr]=audioread(fullfile(folder,d(fi).name));
            if size(a,2)>1,a=mean(a,2);end
            a=double(a(:));if sr~=fs,a=resample(a,fs,sr);end
            a=a-mean(a);pk=max(abs(a));if pk<1e-5,continue;end
            a=a/pk;if length(a)<winSamples,a=repmat(a,ceil(winSamples/length(a)),1);end
            wins=[wins,single(a(1:winSamples))]; %#ok<AGROW>
        catch;end
    end
end
if isempty(wins), wins=synth_wins_local(type,fs,winSamples,maxW); end
end
