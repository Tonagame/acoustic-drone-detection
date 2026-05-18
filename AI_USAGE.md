# How AI Helped in This Project

AI was used as an engineering assistant throughout this acoustic drone detection project. It did not replace the core engineering decisions; it helped accelerate iteration, organize experiments, and explain results.

## Where AI Helped

### 1. Research and Brainstorming

AI helped compare possible detector architectures:

- one generalist CNN,
- five specialist CNNs,
- hybrid generalist + specialist fusion,
- harmonic DSP guard features,
- pretrained pitch/periodicity features,
- learned fusion models.

This helped turn vague ideas into concrete experiments with clear gates: recall, false-alarm rate, SNR behavior, and benchmark conditions.

### 2. Code Generation and Refactoring

AI helped create and organize Python/PyTorch modules for:

- multi-view audio preprocessing,
- CNN training loops,
- FSD50K hard-negative benchmarking,
- harmonic feature extraction,
- learned fusion heads,
- threshold sweeps,
- benchmark plotting,
- passive microphone-array simulation.

The code was iterated locally and tested against the project data and results.

### 3. Debugging and Experiment Tracking

AI helped inspect why some systems failed. The biggest example was the synthetic-noise downfall: models looked strong against synthetic tank/engine sounds but failed on drone audio mixed with real FSD50K vehicle/engine noise.

AI helped reorganize the project around this lesson by separating:

- synthetic tests,
- real hard-negative tests,
- mixed-positive recall tests,
- final false-alarm benchmarks.

### 4. Visualization

AI helped generate benchmark graphs for:

- recall vs false-alarm tradeoffs,
- score index comparisons,
- SNR recall curves,
- synthetic-to-real progress,
- simple harmonic-ladder explanation.

These plots made the engineering story easier to understand and publish.

### 5. Documentation and White Paper

AI helped convert the full project history into an engineering white paper, including:

- trial and error,
- model iterations,
- training settings,
- benchmark results,
- limitations,
- next steps.

The goal was to make the project understandable to both technical reviewers and portfolio readers.

## Important Note

AI was used as a development assistant, not as a source of benchmark truth. Results came from local code, datasets, model checkpoints, and generated benchmark outputs. The project still needs independent validation on real FPV drone recordings and real field noise.
