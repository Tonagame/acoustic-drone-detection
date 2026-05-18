# Phase 1 – Mono Drone Audio Detector

Binary classifier (drone / no\_drone) trained on log-Mel spectrograms using a small CNN.

---

## 1. Dataset – DADS

**Dataset:** Drone Audio Detection Samples (DADS)  
**Source:** [geronimobasso/drone-audio-detection-samples](https://huggingface.co/datasets/geronimobasso/drone-audio-detection-samples) on Hugging Face  

### Download options

**Option A – Hugging Face Python CLI (recommended)**
```bash
pip install huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='geronimobasso/drone-audio-detection-samples',
                  repo_type='dataset', local_dir='hf_dads')
"
```

**Option B – `datasets` library**
```python
from datasets import load_dataset
ds = load_dataset('geronimobasso/drone-audio-detection-samples')
```
Then export the audio files to disk and sort them by label.

### Required folder structure after placement

```
drone_detect/
  data/
    raw/
      drone/          <- put all drone WAV files here
        file001.wav
        file002.wav
        ...
      no_drone/       <- put all background WAV files here
        file001.wav
        file002.wav
        ...
```

Files must be directly inside `drone/` and `no_drone/` (no sub-folders).  
Accepted format: `.wav`, PCM, 16 kHz, mono (the code auto-converts other sample rates and stereo files).

---

## 2. Required MATLAB Toolboxes

| Toolbox | Used for |
|---|---|
| **Audio Toolbox** | `audioDatastore`, `melSpectrogram` |
| **Deep Learning Toolbox** | `trainNetwork`, `convolution2dLayer`, `confusionchart`, etc. |
| **Signal Processing Toolbox** | `resample` (bundled with most MATLAB installs) |

MATLAB R2021a or newer is recommended.

---

## 3. Project layout

```
drone_detect/
  data/
    raw/
      drone/
      no_drone/
    processed/         (reserved for future phases)
  features/
    logmel_features.mat   <- cached feature arrays (created at runtime)
  models/
    drone_cnn_phase1.mat  <- trained network    (created at runtime)
  results/
    phase1_metrics.mat    <- accuracy + per-class stats (created at runtime)
    confusion_chart.png   <- confusion chart image      (created at runtime)
  src/
    prepare_dataset.m   load & preprocess WAV files, split by file
    extract_logmel.m    compute log-Mel spectrograms
    train_drone_cnn.m   build & train CNN
    evaluate_model.m    compute metrics, save results
    run_phase1.m        orchestration script (entry point)
  README_phase1.md
```

---

## 4. Running the pipeline

Open MATLAB, `cd` to the `drone_detect/` root, then:

```matlab
run("src/run_phase1.m")
```

The script runs all four steps automatically and prints progress to the Command Window.

---

## 5. Pipeline stages

### 5.1 Dataset preparation (`prepare_dataset.m`)
- Loads WAV files via `audioDatastore` with `'LabelSource', 'foldernames'`.
- Converts to mono and resamples to 16 000 Hz if necessary.
- Normalises amplitude per file (peak normalisation).
- Slices each file into **1-second windows** with **50% overlap** (hop = 0.5 s).
- Splits by **file** (not by window) into 70 / 15 / 15 % to prevent data leakage.

### 5.2 Feature extraction (`extract_logmel.m`)
Each 1-second window is converted to a log-Mel spectrogram:

| Parameter | Value |
|---|---|
| Sample rate | 16 000 Hz |
| Window length | 400 samples (25 ms) |
| Overlap length | 240 samples (hop = 160, ~10 ms) |
| FFT length | 512 |
| Mel bands | 64 |
| Scaling | log10(S + eps) |

Output shape per window: **64 × 98 × 1** (bands × frames × channels).  
Features are cached to `features/logmel_features.mat` for faster re-runs.

### 5.3 CNN training (`train_drone_cnn.m`)

| Layer | Details |
|---|---|
| Input | 64 × 98 × 1, z-score normalisation |
| Conv1 + BN + ReLU + MaxPool | 16 filters, 3×3, pool 2×2 |
| Conv2 + BN + ReLU + MaxPool | 32 filters, 3×3, pool 2×2 |
| Conv3 + BN + ReLU | 64 filters, 3×3 |
| GlobalAveragePooling | → 64-dim vector |
| FC(2) + Softmax + Classification | binary output |

Training: Adam, 30 epochs, lr = 1e-3 (halved every 10 epochs), batch 64.

### 5.4 Evaluation (`evaluate_model.m`)
Metrics on the held-out test set:
- Overall accuracy
- Per-class: Precision, Recall, False-Positive Rate, False-Negative Rate
- Confusion chart saved as `results/confusion_chart.png`

---

## 6. Outputs

| File | Description |
|---|---|
| `models/drone_cnn_phase1.mat` | Trained `SeriesNetwork` object |
| `features/logmel_features.mat` | Cached feature arrays (train / val / test) |
| `results/phase1_metrics.mat` | Struct with accuracy, precision, recall, FPR, FNR, confusion matrix |
| `results/confusion_chart.png` | Normalised confusion chart |

---

## 7. Memory note

With the full 180 k-row DADS dataset, feature extraction can require several GB of RAM.  
If you run out of memory, use a subset of files during initial experimentation by limiting the files placed in `data/raw/`.

---

## 8. What is NOT in Phase 1

- Microphone-array beamforming
- Noise augmentation / mixing
- Simulink integration
- Raspberry Pi deployment
