function metrics = evaluate_model(net, testFeatures, resultsDir)
% EVALUATE_MODEL  Classify test set and report per-class performance metrics.
%
%   metrics = evaluate_model(net, testFeatures, resultsDir)
%
%   Inputs:
%     net              - trained SeriesNetwork
%     testFeatures.X   - [H x W x 1 x N_test] single array
%     testFeatures.labels - N_test x 1 categorical
%     resultsDir       - folder to write results files
%
%   Outputs (saved to resultsDir):
%     phase1_metrics.mat   - metrics struct
%     confusion_chart.png  - confusion chart image
%
%   Console output:
%     Accuracy, per-class Precision / Recall / FPR / FNR

if ~isfolder(resultsDir)
    mkdir(resultsDir);
end

% ------------------------------------------------------------------
% Inference
% ------------------------------------------------------------------
fprintf('Running inference on %d test windows...\n', size(testFeatures.X, 4));
predLabels = classify(net, testFeatures.X);
trueLabels = testFeatures.labels;

% ------------------------------------------------------------------
% Overall accuracy
% ------------------------------------------------------------------
accuracy = mean(predLabels == trueLabels);
fprintf('\n=== Test-Set Evaluation ===\n');
fprintf('Overall accuracy : %.4f  (%.2f%%)\n', accuracy, accuracy * 100);

% ------------------------------------------------------------------
% Confusion matrix (raw counts)
% ------------------------------------------------------------------
classNames = categories(trueLabels);
C          = confusionmat(trueLabels, predLabels, 'Order', classNames);

% ------------------------------------------------------------------
% Per-class precision / recall / FPR / FNR
% ------------------------------------------------------------------
nClasses   = numel(classNames);
precision  = zeros(1, nClasses);
recall     = zeros(1, nClasses);
FPR        = zeros(1, nClasses);
FNR        = zeros(1, nClasses);

fprintf('\n%-12s  Precision  Recall    FPR       FNR\n', 'Class');
fprintf('%s\n', repmat('-', 1, 56));

for k = 1 : nClasses
    TP = C(k, k);
    FP = sum(C(:, k)) - TP;
    FN = sum(C(k, :)) - TP;
    TN = sum(C(:))    - TP - FP - FN;

    precision(k) = safeDiv(TP, TP + FP);
    recall(k)    = safeDiv(TP, TP + FN);
    FPR(k)       = safeDiv(FP, FP + TN);
    FNR(k)       = safeDiv(FN, FN + TP);

    fprintf('%-12s  %.4f     %.4f    %.4f    %.4f\n', ...
        classNames{k}, precision(k), recall(k), FPR(k), FNR(k));
end
fprintf('%s\n', repmat('-', 1, 56));

% ------------------------------------------------------------------
% Build metrics struct
% ------------------------------------------------------------------
metrics.accuracy        = accuracy;
metrics.classNames      = classNames;
metrics.confusionMatrix = C;
metrics.precision       = precision;
metrics.recall          = recall;
metrics.FPR             = FPR;
metrics.FNR             = FNR;

% ------------------------------------------------------------------
% Confusion chart → PNG
% ------------------------------------------------------------------
fig = figure('Visible', 'off', 'Position', [100 100 560 480]);
cm  = confusionchart(trueLabels, predLabels, ...
    'ClassNames', classNames, ...
    'Title',      'Drone vs No-Drone  –  Test Set');
cm.RowSummary    = 'row-normalized';
cm.ColumnSummary = 'column-normalized';

chartPath = fullfile(resultsDir, 'confusion_chart.png');
exportgraphics(fig, chartPath, 'Resolution', 150);
close(fig);
fprintf('\nConfusion chart saved : %s\n', chartPath);

% ------------------------------------------------------------------
% Save metrics
% ------------------------------------------------------------------
metricsPath = fullfile(resultsDir, 'phase1_metrics.mat');
save(metricsPath, 'metrics');
fprintf('Metrics saved         : %s\n', metricsPath);
end


% =========================================================================
function v = safeDiv(num, den)
% Division that returns 0 when denominator is zero.
if den == 0
    v = 0;
else
    v = num / den;
end
end
