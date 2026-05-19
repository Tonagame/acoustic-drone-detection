# Code

This folder contains the project code only. Large datasets, trained model checkpoints, and generated caches are intentionally not included in the GitHub repository.

## Main Folders

| Folder | Purpose |
|---|---|
| `src/phase3_mid_fusion/` | Current best direction: mid-fusion v1 and guard-neck fusion v2 |
| `src/phase3_real_noise_specialists/` | Five-specialist real-noise CNN ensemble |
| `src/phase2_harmonic_fusion/` | CNN latent + harmonic DSP fusion |
| `src/phase2b_pitch_guard/` | Pretrained pitch/periodicity features + learned fusion |
| `src/phase2v5_real_noise/` | Real-noise generalist CNN training and benchmark code |
| `src/phase3_array/` | Passive microphone-array / beamforming simulation tools |
| `src/fsd50k_hard_negative_eval/` | Real-noise FSD50K benchmark tools |
| `scripts/` | Older training, comparison, and live-detector scripts kept for traceability |

## Current Experiment Commands

Because the Python package root is inside `code/`, run commands from the repository root with:

```powershell
$env:PYTHONPATH="code"
```

Train Guard-Neck Mid Fusion v2:

```powershell
python -m src.phase3_mid_fusion.train_guard_neck_fusion
```

Benchmark Guard-Neck Mid Fusion v2:

```powershell
python -m src.phase3_mid_fusion.benchmark_guard_neck_fusion --guard-neck-threshold 0.55
```

Run the five-specialist raw-vs-five ablation:

```powershell
python -m src.phase3_real_noise_specialists.ablate_raw_vs_five --windows-per-condition 250
```
