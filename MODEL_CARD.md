# Model Card: Acoustic Drone Detection Prototype

## Model Family

Passive acoustic drone detection using:

- multi-view log-mel audio preprocessing,
- CNN binary classifiers,
- five specialist CNNs,
- harmonic DSP features,
- pretrained pitch/periodicity features,
- learned ML fusion.

## Intended Use

Research and engineering evaluation for passive acoustic drone detection. The current system is intended for offline benchmarking and prototype development, not operational deployment.

## Input

- Mono audio windows for detector models.
- 16 kHz sample rate in the current pipeline.
- 1-second windows.
- Five spectral views: raw, HPF-150, HPF-250, BPF-200-6000, BPF-500-6000.

## Output

- Drone probability.
- Drone/no-drone decision after thresholding and optional temporal smoothing.

## Best Current Benchmark

| Metric | Result |
|---|---:|
| Clean drone recall | 99.20% |
| Mixed drone + real FSD50K noise recall | 91.05% |
| -20 dB mixed recall | 48.40% |
| -15 dB mixed recall | 85.60% |
| -10 dB mixed recall | 98.40% |
| -5 dB mixed recall | 99.20% |
| False alarm rate on benchmark negatives | 0.00% |

## Limitations

The system has not yet been validated on:

- real FPV drone recordings,
- real tank or battlefield vehicle recordings,
- real microphone-array field recordings,
- 48 kHz high-frequency FPV audio,
- embedded real-time hardware.

The current 16 kHz pipeline cannot analyze frequencies above about 8 kHz.

## Safety and Responsible Use

This repository is for acoustic detection research and engineering portfolio demonstration. It should not be presented as an operational military targeting system. Real-world deployment would require legal, safety, privacy, and field-validation review.
