# External Audio Datasets

This folder is for large third-party datasets used as negative/background data.

## FSD50K

Official source: https://zenodo.org/records/4060432

Local folder: `data/external/FSD50K/`

Downloaded first:

- `FSD50K.doc.zip`
- `FSD50K.metadata.zip`
- `FSD50K.ground_truth.zip`
- full dev/eval audio split archives
- extracted WAV folders under `FSD50K/extracted/`

Full audio is optional and large. Use:

```powershell
.\tools\download_external_audio_data.ps1 -FSD50KAudio
```

Current local useful files:

- `FSD50K/extracted/FSD50K.dev_audio/`
- `FSD50K/extracted/FSD50K.eval_audio/`
- `FSD50K/fsd50k_vehicle_label_counts.csv`
- `FSD50K/fsd50k_vehicle_engine_candidates.csv`

The candidate CSV includes local paths for labels such as `Engine`, `Vehicle`,
`Truck`, `Car`, `Bus`, `Motorcycle`, `Aircraft`, `Explosion`, and
`Gunshot_and_gunfire`.

## MAD

Official paper/repo:

- https://www.nature.com/articles/s41597-024-03511-w
- https://github.com/kaen2891/military_audio_dataset
- Kaggle: https://www.kaggle.com/datasets/junewookim/mad-dataset-military-audio-dataset

Local folder: `data/external/MAD/`

The GitHub metadata was downloaded locally. The full audio is hosted on Kaggle
and requires Kaggle API credentials.

Current local MAD files:

- `MAD/README.md`
- `MAD/mad_dataset_annotation.csv`
- `MAD/training.csv`
- `MAD/test.csv`
- `MAD/youtube_audio_download.py`

MAD label mapping from the paper:

- `0`: communication
- `1`: gunshot
- `2`: footsteps
- `3`: shelling
- `4`: vehicle
- `5`: helicopter
- `6`: fighter

Expected credential path on Windows:

```text
C:\Users\Haim\.kaggle\kaggle.json
```

After credentials are available:

```powershell
C:\Users\Haim\miniconda3\Scripts\kaggle.exe datasets download -d junewookim/mad-dataset-military-audio-dataset -p data\external\MAD --unzip
```
