# Drone Detection Project Status And Next Phases

Date: 2026-05-17

## Current System

The current best detector is the Hybrid Option2+Option3 smoothed system.

Option 2 is one generalist CNN trained with five audio views:

1. raw
2. highpass 150 Hz
3. highpass 250 Hz
4. bandpass 200-6000 Hz
5. bandpass 500-6000 Hz

Option 3 is a five-specialist CNN ensemble, one model per audio view.

The Hybrid system combines Option 3 as the sensitive detector and Option 2 as the confirmation / false-alarm guard. It also uses simple veto logic and temporal smoothing.

## What Worked Before

On synthetic/internal condition tests, the Hybrid system was strong:

- Precision around 99.8%
- Recall around 91.7%
- F1 around 95.6%
- Tank false alarms near 0%
- Engine false alarms near 0%
- Crowd false alarms near 0%

This was useful, but most hard negatives were synthetic tank/engine/crowd sounds.

## New Real-Noise Benchmark

We downloaded and extracted FSD50K.

Local FSD50K paths:

- data/external/FSD50K/extracted/FSD50K.dev_audio/
- data/external/FSD50K/extracted/FSD50K.eval_audio/
- data/external/FSD50K/fsd50k_vehicle_engine_candidates.csv

FSD50K real negative labels used:

- Engine
- Engine_starting
- Motor_vehicle_(road)
- Vehicle
- Truck
- Car
- Car_passing_by
- Bus
- Motorcycle
- Aircraft
- Explosion
- Gunshot_and_gunfire

## FSD50K Negative-Only Benchmark

This benchmark measured false alarms on real no-drone FSD50K sounds.

Results:

- Option 2 false alarm: 26.33%
- Option 3 false alarm: 22.23%
- Hybrid false alarm: 3.28%
- Hybrid + Harmonic Guard false alarm: 3.28%

Interpretation:

The Hybrid is much better than Option2 or Option3 alone for false alarms, but 3.28% is still not mission-ready.

Worst false-alarm labels for Hybrid:

- Engine: 8.78%
- Engine_starting: 5.37%
- Car_passing_by: 4.67%
- Motorcycle: 4.05%
- Vehicle: 3.60%
- Car: 2.82%

## FSD50K Mixed Positive Benchmark

This benchmark measured detection rate with real DADS drone mixed with real FSD50K interference.

Drone alone recall:

- Option 2: 92.5%
- Option 3: 97.5%
- Hybrid: 85.0%
- Hybrid + Harmonic Guard: 85.0%

Drone + real FSD50K interference recall:

- Option 2: 37.03%
- Option 3: 30.95%
- Hybrid: 9.32%
- Hybrid + Harmonic Guard: 9.32%

Interpretation:

The system was trained/tuned mostly on synthetic tank/engine noise. It does not generalize well to real vehicle/engine interference. The Hybrid is safe but too conservative under real interference.

## Current Diagnosis

The main problem is a synthetic-to-real gap.

The detector learned synthetic tank/engine patterns, but real FSD50K engine, vehicle, motorcycle, and car-pass sounds have different:

- microphone noise
- reverberation
- compression artifacts
- background mixtures
- nonstationary pass-by behavior
- real broadband clutter
- unexpected harmonic structure

So the next major fix is to train a new model iteration with real FSD50K hard negatives and real FSD50K mixed positives.

## Existing Harmonic Guard

A new experimental harmonic guard exists under:

- src/harmonic_guard/
- results/harmonic_guard/

It is guard-only. It does not modify audio and does not train a model.

It computes:

- low f0 estimate in 30-150 Hz
- harmonic ladder up to 4000 Hz
- harmonicity score
- upper harmonic explained ratio
- low-band vehicle risk score

It is diagnostically useful, but it did not improve FSD50K benchmark results yet. It needs real false-alarm tuning.

## Phase Plan

### Phase 0 - Refactor, No Training

Goal: make future model iterations easier and safer.

Tasks:

- Split CNN into encode() and classify_from_latent().
- Add classify_from_latent() head support.
- Add --sample_rate flag to new training/evaluation scripts.
- Save checkpoint metadata:
  - phase name
  - sample rate
  - view names
  - view filters
  - class mapping
  - training data recipe
  - thresholds
- Add passive latent-saving utility for clean drone data.
- Do not modify old checkpoints.
- Do not change live detector behavior.

Expected time: 2-4 hours.

### Phase 1 - New Generalist, Real-Noise Joint Training

Goal: train one new generalist CNN with real FSD50K noise.

Training data:

Positive:

- DADS drone alone
- DADS drone + FSD50K engine
- DADS drone + FSD50K vehicle
- DADS drone + motorcycle/car/truck/bus
- optional drone + aircraft/explosion/gunfire

Negative:

- FSD50K engine alone
- FSD50K vehicle alone
- FSD50K motorcycle/car/truck/bus alone
- FSD50K aircraft/explosion/gunfire alone
- original no_drone data

Training details:

- One CNN, five-view random filter augmentation.
- Balanced sampling.
- Focal loss or class-balanced cross entropy.
- Save as a new model only. Do not overwrite old models.

Gate:

- FSD50K negative false alarm < 10%
- clean drone recall >= 88-90%
- mixed drone + FSD50K recall meaningfully above current baseline

Expected time: 3-4 hours.

### Phase 1b - Benchmark And Threshold Calibration

Goal: calibrate thresholds after Phase 1.

Benchmarks:

- clean drone recall
- FSD50K negative false alarm
- DADS drone + FSD50K mixed recall
- original synthetic tank/engine/crowd condition tests
- pure noise / wind / speech / crowd false alarms

Decide whether the model should run as:

- replacement generalist
- second-stage confirmation model
- part of a new hybrid

### Phase 2 - Add Harmonic DSP Auxiliary Input

Goal: use harmonic features without destroying drone evidence.

Architecture:

CNN latent vector + harmonic feature vector -> small classifier head

Harmonic features:

- f0_hz
- hps_confidence
- low_band_ratio
- harmonicity_score
- upper_harmonic_explained_ratio
- impulse_score
- vehicle_risk_score

Training:

- Freeze most of Phase 1 backbone.
- Fine-tune upper layers / classifier head only.

Gate:

- vehicle/engine FAR < 5%
- clean recall does not regress
- mixed recall improves or holds

Expected time: 1-2 hours.

### Phase 3 - Five-Specialist Real-Noise Ensemble

Goal: regain sensitivity with real-noise-trained specialists.

Train five CNN specialists:

1. raw
2. HPF-150
3. HPF-250
4. BPF-200-6000
5. BPF-500-6000

Use the same real FSD50K data recipe as Phase 1.

Rebuild hybrid:

- specialists as sensitive front-end
- Phase 2 generalist as false-alarm guard
- temporal smoothing
- harmonic risk as auxiliary guard/fusion signal

Gate:

- FSD50K FAR < 5%
- clean drone recall >= 92%
- mixed drone + FSD50K recall >= 75%

Expected time: overnight.

### Phase 4 - Deploy And Monitor

Goal: run on real operational/site audio.

Tasks:

- Save false-alarm clips.
- Save missed-detection clips when known.
- Log model scores and harmonic features.
- Build a site-specific noise set.

No automatic retraining.

### Phase 5 - Continual Learning Fine-Tuning

Only if Phase 4 reveals recurring site-specific failures.

Tasks:

- Freeze backbone.
- Fine-tune upper layers using new site noise.
- Use saved clean-drone latent replay buffer to avoid forgetting.
- Do not full-retrain unless necessary.

### Phase 6 - 48 kHz FPV Path

Only when real FPV drone data exists.

Reason:

The current 16 kHz detector only hears up to 8 kHz. FPV drone whine may contain useful energy above 8 kHz.

Tasks:

- collect FPV data at 48 kHz
- add high-frequency views
- design 48 kHz model path
- adapt from Phase 3 checkpoint if possible

## Immediate Recommendation

Do Phase 0 first.

Then Phase 1 generalist real-noise training.

Do not train the five-specialist ensemble yet. First prove the FSD50K data recipe fixes the collapse with a faster one-model experiment.
