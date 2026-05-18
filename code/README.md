# Code

This folder contains the project code only. Large datasets, trained model checkpoints, and generated caches are intentionally not included in the GitHub repository.

## Main Folders

| Folder | Purpose |
|---|---|
| `src/phase2v5_real_noise/` | Real-noise generalist CNN training and benchmark code |
| `src/phase2_harmonic_fusion/` | CNN latent + harmonic DSP fusion |
| `src/phase2b_pitch_guard/` | Pretrained pitch/periodicity features + learned fusion |
| `src/phase3_real_noise_specialists/` | Five-specialist real-noise CNN ensemble |
| `src/phase3_array/` | Passive microphone-array / beamforming simulation tools |
| `src/fsd50k_hard_negative_eval/` | Real-noise FSD50K benchmark tools |
| `scripts/` | Older training, comparison, and live-detector scripts kept for traceability |

## Running Modules

Because the Python package root is inside `code/`, run commands from the repository root with:

```bash
set PYTHONPATH=code
python -m src.phase3_real_noise_specialists.plot_iteration_comparison
```

On PowerShell:

```powershell
$env:PYTHONPATH="code"
python -m src.phase3_real_noise_specialists.plot_iteration_comparison
```
