"""
download_dads.py
Read cached Arrow IPC shards and write WAV files to:
  data/raw/drone/     (label == 1)
  data/raw/no_drone/  (label == 0)

Bypasses the datasets audio decoder (torchcodec / FFmpeg) entirely.
Reads raw WAV bytes stored in the Arrow cache and decodes with soundfile.
"""

import io
import os
import glob
import numpy as np
import soundfile as sf
import pyarrow as pa
import pyarrow.ipc as ipc
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
OUT_ROOT   = SCRIPT_DIR / "data" / "raw"
CACHE_DIR  = Path("E:/hf_dataset_cache")

ARROW_DIR  = (CACHE_DIR /
              "geronimobasso___drone-audio-detection-samples" /
              "default" / "0.0.0" /
              "981b832c35b45a57518c989f0f79101adb4ae91f")

LABEL_MAP = {0: "no_drone", 1: "drone"}

# ── Prepare output directories ───────────────────────────────────────────
for folder in LABEL_MAP.values():
    (OUT_ROOT / folder).mkdir(parents=True, exist_ok=True)

# ── Find Arrow shards ────────────────────────────────────────────────────
arrow_files = sorted(ARROW_DIR.glob("*.arrow"))
if not arrow_files:
    raise FileNotFoundError(
        f"No .arrow files found in:\n  {ARROW_DIR}\n"
        "Run the script once with datasets to download the cache first."
    )

print("=" * 60)
print(" DADS  ->  WAV extractor (direct Arrow reader)")
print("=" * 60)
print(f"Arrow shards : {len(arrow_files)}")
print(f"Output root  : {OUT_ROOT}")
print()

# ── Extract ───────────────────────────────────────────────────────────────
counters  = {0: 0, 1: 0}
errors    = 0
global_i  = 0
log_every = 5000

for shard_path in arrow_files:
    shard_name = shard_path.name
    with pa.memory_map(str(shard_path), "r") as src:
        reader = ipc.open_stream(src)
        while True:
            try:
                batch = reader.read_next_batch()
            except StopIteration:
                break

            audio_col = batch.column("audio")
            label_col = batch.column("label")

            for j in range(len(batch)):
                global_i += 1
                try:
                    label_int  = int(label_col[j].as_py())
                    label_name = LABEL_MAP[label_int]
                    audio_bytes = audio_col[j].as_py()["bytes"]

                    if not audio_bytes:
                        errors += 1
                        continue

                    audio_arr, sr = sf.read(io.BytesIO(audio_bytes),
                                            dtype="float32")

                    # Ensure mono
                    if audio_arr.ndim > 1:
                        audio_arr = audio_arr.mean(axis=1)

                    counters[label_int] += 1
                    filename = f"{label_name}_{counters[label_int]:06d}.wav"
                    out_path = OUT_ROOT / label_name / filename

                    sf.write(str(out_path), audio_arr, sr, subtype="PCM_16")

                except Exception as exc:
                    errors += 1
                    if errors <= 10:
                        print(f"  [WARN] Row {global_i}: {exc}", flush=True)
                    continue

                if global_i % log_every == 0:
                    print(
                        f"  {global_i:>7,}  |  "
                        f"drone: {counters[1]:,}  "
                        f"no_drone: {counters[0]:,}  "
                        f"errors: {errors}  "
                        f"[{shard_name}]",
                        flush=True,
                    )

# ── Summary ───────────────────────────────────────────────────────────────
total = counters[0] + counters[1] + errors
print()
print("=" * 60)
print(" Done!")
print(f"   drone    : {counters[1]:,} files  ->  {OUT_ROOT / 'drone'}")
print(f"   no_drone : {counters[0]:,} files  ->  {OUT_ROOT / 'no_drone'}")
if errors:
    print(f"   skipped  : {errors} / {total} rows")
print("=" * 60)
