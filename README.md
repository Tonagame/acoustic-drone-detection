# Acoustic Drone Detection in Real-Noise Environments

Passive acoustic drone detection research prototype using multi-view CNNs, real-noise hard negatives, harmonic DSP features, pretrained pitch/periodicity estimation, and learned ML fusion.

This repository documents an engineering journey from a simple CNN detector to a stronger real-noise benchmark system. The main lesson was that synthetic tank/engine noise was misleading: models that looked strong on synthetic tests failed when tested against real vehicle and engine recordings. The current best approach combines sensitivity from specialist CNNs with guard features that reduce false alarms.

## Current Best Approach

```text
1-second audio window
-> five spectral views
-> five specialist CNN detectors
-> harmonic DSP features
-> pretrained pitch-estimator features
-> learned ML fusion
-> drone / no-drone decision
```

The best benchmarked approach is:

```text
5-specialist CNN + harmonic DSP + pretrained pitch estimator + learned ML fusion
```

## Latest Benchmark Snapshot

| Metric | Result |
|---|---:|
| Clean drone recall | 99.20% |
| Mixed drone + real FSD50K noise recall | 91.05% |
| -20 dB mixed recall | 48.40% |
| -15 dB mixed recall | 85.60% |
| -10 dB mixed recall | 98.40% |
| -5 dB mixed recall | 99.20% |
| False alarm rate on benchmark negatives | 0.00% |

These are benchmark results, not operational claims. The system still needs validation on real FPV drone recordings, real tank/vehicle field recordings, and real microphone-array recordings.

## Key Figures

### Approach Comparison

![Recall vs false alarm](results/phase3_real_noise_specialists/plots/all_iterations_recall_vs_far.png)

### SNR Recall Curves

![SNR recall curves](results/phase3_real_noise_specialists/plots/all_iterations_snr_recall_curves.png)

### Synthetic-to-Real Lesson

![Synthetic vs real progress](results/phase3_real_noise_specialists/plots/synthetic_vs_real_progress.png)

### Simple Harmonics Concept

![Simple harmonic ladder](results/phase2_harmonic_fusion/plots/simple_harmonics_ladder.png)

## Repository Structure

```text
src/
  phase2v5_real_noise/          Real-noise generalist CNN pipeline
  phase2_harmonic_fusion/       Harmonic DSP + CNN latent fusion
  phase2b_pitch_guard/          Pitch/periodicity learned fusion guard
  phase3_real_noise_specialists/ Five-specialist real-noise ensemble
  phase3_array/                 Passive microphone-array beamforming simulator/tools
  hybrid_option2_option3/       Earlier hybrid generalist + specialist system
  harmonic_guard/               Earlier harmonic guard experiments
  fsd50k_hard_negative_eval/    FSD50K benchmark tools

docs/
  acoustic_drone_detection_white_paper_publishable.md
  acoustic_drone_detection_engineering_white_paper.md
  *.docx exports

results/
  Selected benchmark plots only should be committed.
```

## How the System Works

### Five-Specialist CNNs

The same 1-second audio window is transformed into five views:

| View | Purpose |
|---|---|
| Raw | Keeps the full signal |
| High-pass 150 Hz | Reduces low-frequency rumble |
| High-pass 250 Hz | Stronger low-frequency rejection |
| Band-pass 200-6000 Hz | Main drone-relevant acoustic band |
| Band-pass 500-6000 Hz | Higher motor/propeller evidence |

Each view has its own CNN. The models output five drone probabilities. This increases sensitivity because one filtered view may reveal drone evidence that another view hides.

### Harmonic DSP

Engines, tanks, generators, and vehicles often produce harmonic ladders:

```text
f0, 2f0, 3f0, 4f0, ...
```

The harmonic DSP stage estimates low-frequency fundamental and harmonic structure. It does not delete or suppress audio. It produces side-channel features such as f0, harmonicity, low-band energy, and vehicle-risk score.

### Pretrained Pitch Estimator

A pretrained pitch/periodicity estimator adds another view of stable periodic structure. It provides features such as pitch confidence, pitch stability, and low-pitch ratio.

### Learned ML Fusion

The final fusion model receives:

- five CNN probabilities,
- specialist weighted score,
- filtered maximum,
- vote count,
- harmonic DSP features,
- pretrained pitch/periodicity features.

It learns the final drone probability instead of relying only on hand-written rules.

## Main Engineering Lesson

Synthetic tank and engine sounds were useful for prototyping, but they created a false sense of robustness. The turning point was benchmarking against real FSD50K vehicle/engine recordings. That forced the system to move from synthetic hard negatives to real-noise training and real-noise evaluation.

## Documentation

Start here:

- [Publishable white paper](docs/acoustic_drone_detection_white_paper_publishable.md)
- [Engineering white paper](docs/acoustic_drone_detection_engineering_white_paper.md)
- [AI usage in this project](AI_USAGE.md)
- [Data and model notes](DATA.md)
- [Model card](MODEL_CARD.md)

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

CUDA is recommended for training. Inference and plotting can run on CPU for small tests.

## Example Commands

Benchmark the Phase 3 specialist/pitch-harmonic system after local datasets and model paths are configured:

```bash
python -m src.phase3_real_noise_specialists.benchmark_phase3_specialists
```

Generate comparison plots:

```bash
python -m src.phase3_real_noise_specialists.plot_iteration_comparison
```

Run microphone-array simulation tools:

```bash
python -m src.phase3_array.run_phase3_system_sim_3d
```

## What Is Not Included

This repository should not include:

- raw DADS or FSD50K audio,
- large trained model checkpoints,
- private recordings,
- generated feature caches,
- large local benchmark outputs.

Use `DATA.md` for dataset instructions and `MODEL_CARD.md` for model notes.

## Status

Research prototype. Best current benchmark is promising, but the next required step is validation on real FPV drone audio and field-recorded non-drone noise.
