function metrics = evaluate_phase2_v3(net, XTest, YTest, resultsDir)
% EVALUATE_PHASE2_V3  Overall accuracy / precision / recall on held-out test set.
%
%   metrics = evaluate_phase2_v3(net, XTest, YTest, resultsDir)
%
%   INPUT
%     net        - trained MATLAB network (DAGNetwork / SeriesNetwork)
%     XTest      - [H × W × 1 × N] single test features
%     YTest      - [N × 1] categorical test labels
%     resultsDir - output directory (results/phase2_v3/)
%
%   OUTPUT
%     metrics  - struct: accuracy, precision, recall, fpRate, fnRate
%
%   SAVED FILES
%     results/phase2_v3/overall_metrics.mat
%     results/phase2_v3/confusion_chart.png

if ~exist(resultsDir, 'dir'), mkdir(resultsDir); end

fprintf('[Eval] Running overall evaluation  (N=%d) ...\n', size(XTest,4));

% ── Predict ───────────────────────────────────────────────────────────────
YPred   = classify(net, XTest, 'MiniBatchSize', 64);
scores  = predict(net, XTest,  'MiniBatchSize', 64);

% Ensure column vectors
YTest = YTest(:);
YPred = YPred(:);

classes = categories(YTest);
droneIdx = find(strcmp(classes, 'drone'), 1);
if isempty(droneIdx), droneIdx = 1; end

% ── Confusion matrix ──────────────────────────────────────────────────────
C = confusionmat(YTest, YPred, 'Order', classes);

% For binary case: [drone, no_drone]
% C(i,j) = number of true class i predicted as class j
TP = C(droneIdx, droneIdx);   % drone predicted as drone
FN = sum(C(droneIdx, :)) - TP; % drone predicted as no_drone
nodroneIdx = 3 - droneIdx;   % the other index
FP = C(nodroneIdx, droneIdx);  % no_drone predicted as drone
TN = C(nodroneIdx, nodroneIdx);

% ── Metrics ───────────────────────────────────────────────────────────────
accuracy  = (TP + TN) / (TP + FP + FN + TN + eps);
precision = TP / (TP + FP + eps);
recall    = TP / (TP + FN + eps);
fpRate    = FP / (FP + TN + eps);   % false positive rate (on no_drone)
fnRate    = FN / (FN + TP + eps);   % false negative rate (missed drone)
f1        = 2 * precision * recall / (precision + recall + eps);

metrics.accuracy  = accuracy;
metrics.precision = precision;
metrics.recall    = recall;
metrics.fpRate    = fpRate;
metrics.fnRate    = fnRate;
metrics.f1        = f1;
metrics.TP = TP; metrics.FP = FP; metrics.FN = FN; metrics.TN = TN;
metrics.droneScores  = scores(:, droneIdx);
metrics.trueLabels   = YTest;
metrics.predLabels   = YPred;

% ── Print ─────────────────────────────────────────────────────────────────
fprintf('\n  ╔══════════════════════════════════════╗\n');
fprintf(  '  ║   Phase 2v3  –  Test Set Results     ║\n');
fprintf(  '  ╠══════════════════════════════════════╣\n');
fprintf(  '  ║  Accuracy   : %6.2f %%               ║\n', accuracy  * 100);
fprintf(  '  ║  Precision  : %6.2f %%               ║\n', precision * 100);
fprintf(  '  ║  Recall     : %6.2f %%               ║\n', recall    * 100);
fprintf(  '  ║  F1-Score   : %6.2f %%               ║\n', f1        * 100);
fprintf(  '  ║  FP Rate    : %6.2f %%               ║\n', fpRate    * 100);
fprintf(  '  ║  FN Rate    : %6.2f %%               ║\n', fnRate    * 100);
fprintf(  '  ╚══════════════════════════════════════╝\n\n');
fprintf('  Confusion matrix  [rows=true, cols=pred]:\n');
fprintf('           drone  no_drone\n');
fprintf('  drone    %5d  %5d\n', C(droneIdx, :));
fprintf('  no_drone %5d  %5d\n', C(nodroneIdx, :));
fprintf('\n');

% ── Confusion chart (PNG) ─────────────────────────────────────────────────
try
    fig = figure('Visible', 'off');
    cm  = confusionchart(YTest, YPred, 'Title', 'Phase 2v3 Test Confusion');
    sortClasses(cm, {'drone','no_drone'});
    exportgraphics(fig, fullfile(resultsDir, 'confusion_chart.png'), 'Resolution', 150);
    close(fig);
    fprintf('  Confusion chart → results/phase2_v3/confusion_chart.png\n');
catch ME
    warning('evaluate_phase2_v3: could not save confusion chart: %s', ME.message);
end

% ── Save metrics ──────────────────────────────────────────────────────────
save(fullfile(resultsDir, 'overall_metrics.mat'), 'metrics');
fprintf('  Metrics saved → results/phase2_v3/overall_metrics.mat\n\n');

end
