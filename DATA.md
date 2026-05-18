# Data Notes

This repository does not include raw datasets or large audio files.

## Datasets Used

### DADS

Used as the main drone-positive dataset and no-drone baseline dataset.

Expected local layout:

```text
data/raw/drone/
data/raw/no_drone/
```

### FSD50K

Used as real hard negatives and real-noise mixture sources. Candidate labels included engine, vehicle, motor vehicle, truck, car, bus, motorcycle, aircraft, explosion, and gunshot/gunfire.

Expected local layout/configuration depends on the benchmark module under:

```text
src/fsd50k_hard_negative_eval/
src/phase2v5_real_noise/
```

## Why Data Is Not Committed

Raw audio datasets are large and may have license restrictions. They should be downloaded separately from their official sources and placed in local `data/` folders.

The `.gitignore` is configured to avoid committing:

- raw audio,
- extracted external datasets,
- generated feature caches,
- large trained model checkpoints.

## Reproducibility

The code and documentation describe the experiments, but exact reproduction requires local copies of the datasets and trained model checkpoints. If model weights are shared later, they should be distributed through GitHub Releases, cloud storage, or Git LFS rather than committed directly to the repository.
