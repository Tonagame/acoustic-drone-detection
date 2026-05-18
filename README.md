# Acoustic Drone Detection in Real-Noise Environments

Passive acoustic drone detection research prototype built with Python, PyTorch, DSP, and iterative benchmarking.

The goal was simple to state and hard to solve:

```text
detect drone audio
while rejecting engines, vehicles, tanks, crowds, wind, and other hard false alarms
```

The project evolved from a basic CNN into a multi-component detector:

```text
5 specialist CNNs
+ harmonic DSP features
+ pretrained pitch / periodicity features
+ learned ML fusion
```

## Best Current Result

Current best benchmarked system:

```text
5-specialist CNN + harmonic DSP + pretrained pitch estimator + learned ML fusion
```

| Metric | Result |
|---|---:|
| Clean drone recall | 99.20% |
| Mixed drone + real FSD50K noise recall | 91.05% |
| Drone + FSD50K at -20 dB | 48.40% |
| Drone + FSD50K at -15 dB | 85.60% |
| Drone + FSD50K at -10 dB | 98.40% |
| Drone + FSD50K at -5 dB | 99.20% |
| False alarm rate on benchmark negatives | 0.00% |

These are benchmark results, not operational deployment claims. The system still needs validation on real FPV drone recordings, real vehicle/tank field audio, and real microphone-array recordings.

## Why This Project Is Interesting

Most early models looked good until the benchmark became realistic.

The key engineering lesson:

```text
synthetic tank/engine noise was not enough
```

Early systems rejected synthetic tank and engine sounds, but failed when drone audio was mixed with real vehicle/engine audio from FSD50K. That failure changed the project direction from "make the CNN bigger" to "build a better measurement and fusion system."

## Trial and Error

This project is mostly valuable because of the iteration path. Each version exposed a different weakness.

| Iteration | What was tried | What worked | What failed / lesson |
|---|---|---|---|
| Baseline CNN | One CNN on log-mel spectrograms | Proved drone audio was learnable | Not robust against hard negatives |
| Multi-view generalist CNN | One CNN trained across 5 filtered audio views | Stable and lower false alarms | Less sensitive under heavy noise |
| Five-specialist CNNs | One CNN per audio view | More sensitive to weak drone cues | One specialist could false alarm |
| Rule-based hybrid | Specialists for sensitivity + generalist for confirmation | Excellent on original synthetic benchmark | Became too conservative on real mixed noise |
| Synthetic tank/engine testing | Synthetic hard negatives | Useful for prototyping | Created false confidence |
| Real-noise generalist | DADS + FSD50K hard negatives and mixtures | Reduced false alarms | Missed too many mixed drone cases |
| Balanced real-noise model | Better class weights, SNR range, threshold sweep | Better guard model | Still not sensitive enough alone |
| Harmonic DSP fusion | Add f0/harmonic features to CNN latent | Added interpretable engine/vehicle risk | Gain was real but modest |
| Real-noise specialists | Retrain 5 specialists on real-noise recipe | Restored sensitivity | Needed a learned guard/fusion layer |
| Pitch-harmonic ML fusion | Add pretrained pitch features + learned fusion | Best recall/FAR balance | Still needs real FPV validation |

## Main Failure That Changed the Project

The old hybrid looked strong on synthetic tests, but collapsed on drone audio mixed with real FSD50K vehicle/engine noise.

| System | Mixed drone + real-noise recall |
|---|---:|
| Old multi-view generalist CNN | 37.03% |
| Old five-specialist CNN ensemble | 30.95% |
| Old hybrid | 9.32% |

![Synthetic-to-real failure](results/phase3_real_noise_specialists/plots/synthetic_downfall_old_hybrid.png)

This was the turning point. After this, the project moved to real hard negatives, real-noise mixed positives, threshold sweeps, harmonic features, and learned fusion.

## System Architecture

```text
1-second audio window
        |
        v
five filtered audio views
raw / HPF-150 / HPF-250 / BPF-200-6000 / BPF-500-6000
        |
        v
five specialist CNNs
        |
        v
specialist probability features
        |
        +------ harmonic DSP features
        |
        +------ pretrained pitch / periodicity features
        |
        v
learned ML fusion
        |
        v
drone / no-drone
```

## How It Works

### 1. Five Specialist CNNs

The same 1-second audio window is converted into five views:

| View | Purpose |
|---|---|
| Raw | Keeps the full signal |
| High-pass 150 Hz | Reduces low-frequency rumble |
| High-pass 250 Hz | Stronger low-frequency rejection |
| Band-pass 200-6000 Hz | Main drone-relevant acoustic band |
| Band-pass 500-6000 Hz | Higher motor/propeller evidence |

Each view has its own CNN. This gives the detector multiple chances to find drone evidence under different noise conditions.

### 2. Harmonic DSP

Vehicles, engines, tanks, and generators often produce harmonic ladders:

```text
f0, 2f0, 3f0, 4f0, ...
```

![Simple harmonic ladder](results/phase2_harmonic_fusion/plots/simple_harmonics_ladder.png)

The harmonic DSP stage estimates features such as:

- low-frequency fundamental frequency,
- harmonicity,
- low-band energy,
- vehicle-risk score.

It does not delete audio. It gives the fusion model extra evidence.

### 3. Pretrained Pitch Estimator

A pretrained pitch/periodicity model adds another view of stable pitch-like structure.

It contributes features such as:

- pitch confidence,
- pitch stability,
- low-pitch ratio,
- periodic frame ratio.

### 4. Learned ML Fusion

The final fusion model receives:

- five CNN probabilities,
- specialist weighted score,
- filtered maximum,
- vote count,
- harmonic DSP features,
- pretrained pitch features.

It learns how to combine the evidence instead of relying only on hand-written rules.

## Benchmark Graphs

### Recall vs False Alarm

![Recall vs false alarm](results/phase3_real_noise_specialists/plots/all_iterations_recall_vs_far.png)

### Score Index

![Score index](results/phase3_real_noise_specialists/plots/all_iterations_score_index.png)

### Recall Across SNR

![SNR recall curves](results/phase3_real_noise_specialists/plots/all_iterations_snr_recall_curves.png)

### Synthetic vs Real-Noise Progress

![Synthetic vs real progress](results/phase3_real_noise_specialists/plots/synthetic_vs_real_progress.png)

## Code Structure

```text
src/
  phase2v5_real_noise/            Real-noise generalist CNN pipeline
  phase2_harmonic_fusion/         Harmonic DSP + CNN latent fusion
  phase2b_pitch_guard/            Pretrained pitch + learned fusion guard
  phase3_real_noise_specialists/  Five-specialist real-noise ensemble
  phase3_array/                   Passive microphone-array simulation/tools
  fsd50k_hard_negative_eval/      Real-noise benchmark tools
  hybrid_option2_option3/         Earlier hybrid experiments
  harmonic_guard/                 Earlier harmonic guard experiments

docs/
  acoustic_drone_detection_white_paper_publishable.md
  acoustic_drone_detection_engineering_white_paper.md
  cv_project_summary_he.md

results/
  selected benchmark plots
```

## Documentation

- [Publishable white paper](docs/acoustic_drone_detection_white_paper_publishable.md)
- [Engineering white paper](docs/acoustic_drone_detection_engineering_white_paper.md)
- [How AI helped in this project](AI_USAGE.md)
- [Data notes](DATA.md)
- [Model card](MODEL_CARD.md)
- [Hebrew CV project summary](docs/cv_project_summary_he.md)

## How AI Helped

AI was used as an engineering assistant for:

- brainstorming architecture options,
- turning ideas into concrete experiments,
- generating and refactoring Python/PyTorch code,
- debugging benchmark failures,
- creating plots and documentation,
- writing the engineering white paper.

The benchmark results came from local experiments, datasets, model checkpoints, and evaluation scripts. AI helped accelerate the engineering process; it was not used as a substitute for measured results.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

CUDA is recommended for training. Some plotting and inference utilities can run on CPU.

## Example Commands

Generate approach comparison plots:

```bash
python -m src.phase3_real_noise_specialists.plot_iteration_comparison
```

Benchmark the latest specialist/fusion system after local datasets and model paths are configured:

```bash
python -m src.phase3_real_noise_specialists.benchmark_phase3_specialists
```

Run the microphone-array system simulator:

```bash
python -m src.phase3_array.run_phase3_system_sim_3d
```

## Data and Model Availability

This repository intentionally does not include:

- raw DADS audio,
- raw FSD50K audio,
- private recordings,
- large trained model checkpoints,
- generated feature caches.

See [DATA.md](DATA.md) and [MODEL_CARD.md](MODEL_CARD.md).

## Current Status

Research prototype.

The best benchmark result is promising, but the next required step is validation on:

- real FPV drone audio,
- real field-recorded vehicle/tank noise,
- real microphone-array recordings,
- 48 kHz high-frequency drone audio.
