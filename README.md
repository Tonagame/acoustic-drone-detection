# Acoustic Drone Detection

Passive acoustic drone detection prototype built with Python, PyTorch, DSP, and real-noise benchmarking.

This repository is organized for portfolio review:

```text
code/              source code
results/graphs/    selected headline graphs
trial_and_error/   engineering story with older graphs
```

Raw datasets, trained model weights, private recordings, and large generated outputs are not included.

## Best Current Approach

```text
Guard-Neck Mid Fusion v2
```

The best current detector uses:

- five specialist CNN encoders,
- five filtered audio views,
- harmonic DSP guard features,
- vehicle-risk features,
- learned mid-fusion at the network neck.

Instead of making five CNN decisions and then applying a hard rule afterward, the system joins the internal CNN feature vectors first, injects the guard features at that join point, and learns the final decision.

```text
audio window
   |
   v
5 filtered views
raw / HPF-150 / HPF-250 / BPF-200-6000 / BPF-500-6000
   |
   v
5 frozen specialist CNN encoders
   |
   v
5 x 64 latent vectors = 320 features
   |
   + harmonic/vehicle guard features
   |   phase2_guard_score
   |   vehicle_risk_score
   |   f0_norm
   |   harmonicity_score
   v
324-dim neck vector
   |
   v
learned MLP fusion head
   |
   v
drone probability
```

## Current Benchmark Snapshot

The recommended operating point is:

```text
Guard-Neck Mid Fusion v2 threshold = 0.55
```

| Metric | Result |
|---|---:|
| Clean drone recall | 99.20% |
| Overall positive recall | 90.10% |
| False alarm rate | 0.31% |
| Precision | 99.67% |
| F1 | 94.64% |
| Drone + FSD50K at -20 dB | 47.60% |
| Drone + FSD50K at -15 dB | 80.40% |
| Drone + FSD50K at -10 dB | 96.80% |

These are benchmark results, not operational deployment claims. The system still needs validation on real FPV drone recordings, real field vehicle noise, and real microphone-array recordings.

![Guard-Neck Mid Fusion comparison](results/graphs/guard_neck_v2_comparison.png)

## Why Mid Fusion

The earlier system used late fusion:

```text
CNN probabilities + guard score + rules -> final decision
```

That was useful for debugging, but it had a weakness: the harmonic/vehicle guard could reject weak drones too aggressively. Drones and vehicles both have harmonic structure, so a hard guard can mistake a low-SNR drone for vehicle-like periodic noise.

Mid fusion was introduced to solve that:

```text
internal CNN features + guard evidence -> learned decision
```

This lets the model learn when harmonic evidence means "vehicle" and when it is still compatible with "drone." The guard becomes evidence, not a hard veto.

## Main Result

| System | Recall | FAR | Precision | F1 |
|---|---:|---:|---:|---:|
| Mid Fusion v1 | 88.25% | 0.46% | 99.49% | 93.53% |
| Guard-Neck Mid Fusion v2 tuned | 90.10% | 0.31% | 99.67% | 94.64% |
| Guard-Neck Mid Fusion v2 sensitive | 95.15% | 2.94% | 97.09% | 96.11% |
| Five-specialist rule | 85.35% | 0.36% | 99.59% | 91.92% |
| Hard harmonic guard | 81.90% | 0.21% | 99.76% | 89.95% |
| Soft harmonic guard | 86.55% | 0.88% | 99.03% | 92.37% |

![Guard-Neck vs Mid Fusion](results/graphs/guard_neck_vs_mid_fusion.png)

## The Important Failure

Early versions looked strong against synthetic tank and engine noise. Then real FSD50K vehicle and engine recordings exposed the problem:

```text
synthetic noise success did not transfer to real noise
```

![Synthetic-to-real failure](results/graphs/synthetic_downfall_old_hybrid.png)

That failure changed the project. The detector moved from synthetic-only hard negatives to real FSD50K hard negatives and real mixed-positive benchmarks.

## Trial And Error

The project history is intentionally documented because the failures are the engineering value:

[Trial and Error Story](trial_and_error/README.md)

Short version:

| Step | Approach | Lesson |
|---|---|---|
| Basic CNN | Log-mel CNN | Clean accuracy was not enough |
| Multi-view generalist | One CNN across five filters | Stable, but missed weak drones |
| Five specialists | One CNN per view | Sensitive, but needed a guard |
| Rule hybrid | Late fusion with hard rules | Good on synthetic, too conservative on real noise |
| Real-noise training | FSD50K negatives and mixtures | Real hard negatives changed everything |
| Harmonic guard | DSP vehicle-risk features | Useful, but hard vetoes hurt recall |
| Mid fusion v1 | Join CNN latents before decision | Better evidence mixing |
| Guard-neck v2 | Inject guard features at the neck | Best balanced benchmark so far |

## Code Map

| Path | Purpose |
|---|---|
| `code/src/phase3_mid_fusion/` | Current mid-fusion and guard-neck fusion experiments |
| `code/src/phase3_real_noise_specialists/` | Five-specialist CNN ensemble |
| `code/src/phase2_harmonic_fusion/` | Harmonic DSP + CNN latent fusion |
| `code/src/phase2b_pitch_guard/` | Pretrained pitch/periodicity features + learned fusion |
| `code/src/phase2v5_real_noise/` | Real-noise generalist CNN |
| `code/src/phase3_array/` | Passive microphone-array / beamforming simulation |
| `code/src/fsd50k_hard_negative_eval/` | Real-noise FSD50K benchmark tools |

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

For module commands, set the package path:

```powershell
$env:PYTHONPATH="code"
```

Example:

```powershell
python -m src.phase3_mid_fusion.benchmark_guard_neck_fusion
```

## How AI Helped

AI was used as an engineering assistant for:

- brainstorming architecture changes,
- refactoring code,
- creating benchmark scripts,
- debugging training and evaluation issues,
- generating graphs,
- writing clear project explanations.

The measured results came from local benchmark scripts. AI accelerated the workflow, but every reported number was produced by running code locally.

## Not Included

The repository intentionally excludes:

- raw DADS audio,
- raw FSD50K audio,
- trained model checkpoints,
- private recordings,
- generated feature caches,
- full benchmark logs.

This keeps the GitHub project readable and lightweight.
