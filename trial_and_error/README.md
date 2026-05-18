# Trial and Error Story

This project did not become good in one clean jump. The useful part of the work was the iteration: each model exposed a different weakness, and each failure forced a better design.

The final direction was:

```text
5-specialist CNN
+ harmonic DSP
+ pretrained pitch estimator
+ learned ML fusion
```

But that was not where the project started.

## The Problem

The task was to detect drone audio while rejecting difficult non-drone sounds:

- engines,
- vehicles,
- tanks,
- generators,
- crowds,
- wind,
- speech,
- general environmental noise.

The hard part was not clean drone detection. The hard part was avoiding false alarms while still detecting drones mixed with real noise.

## Iteration Summary

| Step | Approach | What happened | Lesson |
|---|---|---|---|
| 1 | Basic CNN | Learned drone/no-drone from log-mel spectrograms | Clean accuracy was not enough |
| 2 | Multi-view generalist CNN | One CNN learned five filtered views | Stable, but missed weak drones |
| 3 | Five specialist CNNs | One CNN per filtered view | More sensitive, but easier to false alarm |
| 4 | Rule-based hybrid | Specialists + generalist + smoothing | Excellent on synthetic benchmark |
| 5 | Real FSD50K benchmark | Old systems collapsed on real mixed noise | Synthetic noise was misleading |
| 6 | Real-noise generalist | Trained with FSD50K hard negatives | Low false alarms, poor mixed recall |
| 7 | Balanced real-noise model | Better class weights and threshold sweeps | Better guard, still not enough |
| 8 | Harmonic DSP fusion | Added engine/vehicle harmonic features | Helpful but modest |
| 9 | Real-noise specialists | Retrained specialists with real noise | Sensitivity returned |
| 10 | Pitch-harmonic ML fusion | Added pretrained pitch features and learned fusion | Best current benchmark |

## Step 1: Basic CNN

The first detector was a simple CNN on log-mel spectrograms.

Pipeline:

```text
audio window
-> log-mel spectrogram
-> CNN
-> drone / no-drone
```

This proved that drone audio could be learned, but it was not robust enough. A clean test score did not mean the detector could survive engines, vehicles, or crowd noise.

Lesson:

```text
basic CNN detection works, but hard negatives decide whether it is useful
```

## Step 2: Multi-View Generalist CNN

The next idea was to show the same audio through multiple frequency views:

```text
raw
high-pass 150 Hz
high-pass 250 Hz
band-pass 200-6000 Hz
band-pass 500-6000 Hz
```

One CNN was trained across all views. During training, it saw random filtered versions of the same type of sample. During inference, it ran on all five views and combined the scores.

What improved:

- false alarms became lower,
- the model became more stable,
- filtered views helped reduce low-frequency clutter.

What failed:

- one generalist CNN became conservative,
- weak drones under real noise were still missed.

Lesson:

```text
one model can be stable, but stability can cost sensitivity
```

## Step 3: Five Specialist CNNs

The next idea was to train five separate CNNs:

| Specialist | Input |
|---|---|
| raw specialist | raw audio |
| HPF-150 specialist | high-pass 150 Hz |
| HPF-250 specialist | high-pass 250 Hz |
| BPF-200-6000 specialist | band-pass 200-6000 Hz |
| BPF-500-6000 specialist | band-pass 500-6000 Hz |

This made the system more sensitive because each model could specialize in one spectral view.

What improved:

- better weak-drone sensitivity,
- different views could catch different drone cues.

What failed:

- if one specialist fired incorrectly, the system could false alarm,
- specialist sensitivity needed a guard.

Lesson:

```text
specialists are good at finding weak evidence, but they need confirmation
```

## Step 4: Rule-Based Hybrid

The project then combined:

```text
five-specialist CNNs
+ multi-view generalist CNN
+ hand-written fusion rules
+ simple veto
+ temporal smoothing
```

The specialists acted as a sensitive front end. The generalist acted as a confirmation model. Temporal smoothing reduced one-window spikes.

This looked excellent on the early benchmark.

Lesson at the time:

```text
hybrid architecture looked like the right direction
```

But the benchmark was still not realistic enough.

## Step 5: The Synthetic Noise Downfall

This was the turning point.

The old hybrid looked strong against synthetic tank and engine sounds. But when tested with real FSD50K vehicle and engine recordings, performance collapsed.

| System | Mixed drone + real-noise recall |
|---|---:|
| Old multi-view generalist CNN | 37.03% |
| Old five-specialist CNN ensemble | 30.95% |
| Old hybrid | 9.32% |

![Synthetic-to-real failure](graphs/synthetic_downfall_old_hybrid.png)

Why this mattered:

- synthetic tank/engine noise was too simple,
- the model learned the synthetic distribution,
- rejecting synthetic noise did not prove real-world robustness,
- the hybrid became too conservative when real vehicle noise was mixed with drone audio.

Main lesson:

```text
synthetic hard negatives are useful for prototyping, but real hard negatives are required for trust
```

## Step 6: Real-Noise Generalist

The next model was trained with:

- DADS drone audio,
- DADS no-drone audio,
- FSD50K real vehicle/engine hard negatives,
- drone + FSD50K noise mixtures.

What improved:

- false alarms dropped,
- real nuisance audio was handled better.

What failed:

- the model became too conservative,
- it rejected many drone + real-noise mixtures.

Lesson:

```text
false-alarm control can destroy recall if the model becomes too cautious
```

## Step 7: Balanced Real-Noise Training

The training recipe was adjusted:

- stronger mixed-positive exposure,
- different class weights,
- harder SNR ranges,
- threshold sweeps after training.

This produced a better guard model, but it still was not the final detector. A single generalist model could not carry the whole task.

Lesson:

```text
thresholds and class weights matter, but architecture still matters more
```

## Step 8: Harmonic DSP Fusion

Engines, vehicles, tanks, and generators often create harmonic ladders:

```text
f0, 2f0, 3f0, 4f0...
```

![Simple harmonics](graphs/simple_harmonics_ladder.png)

The project added harmonic DSP features:

- low-frequency fundamental estimate,
- harmonicity,
- low-band energy,
- upper harmonic structure,
- vehicle-risk score.

Important design choice:

```text
do not delete audio before the CNN
```

Instead, harmonic evidence was passed as side-channel features to a fusion model.

The harmonic fusion model improved some mixed-noise conditions while keeping false alarms controlled.

![Phase 2 positive recall](graphs/phase2_vs_v5c_positive_recall.png)

![Phase 2 false alarms](graphs/phase2_vs_v5c_false_alarms.png)

Threshold choice became important:

![Phase 2 threshold tradeoff](graphs/phase2_threshold_tradeoff.png)

Score distributions helped show how the fusion model separated conditions:

![Phase 2 score distribution](graphs/phase2_score_distribution.png)

Lesson:

```text
harmonic features help, but they should guide fusion, not directly erase sound
```

## Step 9: Real-Noise Specialists

The five-specialist idea was brought back, but this time with the real-noise training recipe.

This combined the best ideas so far:

- specialist sensitivity,
- real FSD50K hard negatives,
- real mixed-positive examples,
- better benchmark discipline.

What improved:

- sensitivity returned,
- the model was less dependent on synthetic assumptions.

What still needed work:

- specialist outputs still needed a smart final decision layer.

Lesson:

```text
specialists are strongest when trained on the noise they will actually face
```

## Step 10: Pitch-Harmonic Learned Fusion

The final improvement added a pretrained pitch / periodicity estimator and a learned fusion model.

The fusion model received:

- five specialist CNN probabilities,
- specialist weighted score,
- filtered maximum,
- vote count,
- harmonic DSP features,
- pretrained pitch features.

Instead of a fixed rule, the system learned how to combine evidence.

Final result:

| Metric | Result |
|---|---:|
| Clean drone recall | 99.20% |
| Mixed drone + real FSD50K noise recall | 91.05% |
| False alarm rate on benchmark negatives | 0.00% |

![All approaches recall vs false alarm](graphs/all_iterations_recall_vs_far.png)

![All approaches score index](graphs/all_iterations_score_index.png)

SNR behavior showed the remaining weakness:

![SNR recall curves](graphs/all_iterations_snr_recall_curves.png)

At -20 dB, detection is still difficult. At -15 dB and above, the final system is much stronger.

Progress summary:

![Synthetic vs real progress](graphs/synthetic_vs_real_progress.png)

## Final Engineering Lesson

The main lesson was not "use a bigger CNN."

The main lesson was:

```text
build the benchmark correctly,
then build the model around the failure modes
```

The final system works better because each component has a job:

| Component | Job |
|---|---|
| Five specialist CNNs | Sensitivity to weak drone evidence |
| Harmonic DSP | Interpretable engine/vehicle risk evidence |
| Pretrained pitch estimator | Learned periodicity evidence |
| Learned ML fusion | Final decision from all signals |

## Current Limitations

The system is still a research prototype.

It still needs:

- real FPV drone recordings,
- real tank / vehicle field recordings,
- real microphone-array recordings,
- 48 kHz high-frequency drone testing,
- deployment testing on live hardware.

## Best Current Summary

```text
5-specialist CNN + harmonic DSP + pretrained pitch estimator + learned ML fusion
```

Best current benchmark:

```text
91.05% mixed real-noise recall
0.00% false alarm rate on benchmark negatives
```

This is the best current project direction, but not yet a final operational detector.
