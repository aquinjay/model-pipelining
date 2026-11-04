"""Generate two reproducible daily series using only the Python standard library."""

import argparse
import csv
import hashlib
import json
import random
import sys
from datetime import date, timedelta
from pathlib import Path


def generate(output_dir: Path) -> None:
    seed = 42
    count = 200
    start = date(2020, 1, 1)
    mean = 10.0
    noise_stddev = 1.0
    shift_index = 100
    shift_size = 5.0
    filenames = ("constant_mean.csv", "mean_shift.csv", "manifest.json")
    if any((output_dir / name).exists() for name in filenames):
        raise FileExistsError(f"Refusing to overwrite an existing dataset in {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    noise = [rng.gauss(0.0, noise_stddev) for _ in range(count)]
    datasets = []
    for filename, shift in ((filenames[0], 0.0), (filenames[1], shift_size)):
        path = output_dir / filename
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(("timestamp", "value"))
            for index, error in enumerate(noise):
                timestamp = f"{start + timedelta(days=index)}T00:00:00Z"
                value = mean + error + (shift if index >= shift_index else 0.0)
                writer.writerow((timestamp, format(value, ".17g")))
        datasets.append({
            "file": filename,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "initial_mean": mean,
            "mean_shift": shift,
            "shift_start_timestamp": (
                f"{start + timedelta(days=shift_index)}T00:00:00Z" if shift else None
            ),
        })

    manifest = {
        "dataset_version": 1,
        "generator": "scripts/generate_synthetic.py",
        "python_version": sys.version.split()[0],
        "seed": seed,
        "random_generator": "random.Random(seed).gauss(0.0, noise_stddev)",
        "observations_per_series": count,
        "start_timestamp": f"{start}T00:00:00Z",
        "frequency": "daily, including weekends",
        "units": "arbitrary synthetic units",
        "noise_stddev": noise_stddev,
        "process": "value[t] = initial_mean + mean_shift * I[t >= 100] + noise[t]",
        "shared_noise": "Both scenarios use the same noise sequence for controlled comparison.",
        "availability": "Each value is available at its timestamp, with no revisions or delay.",
        "forecast_convention": "Issue after observing value[t]; predict value[t+1].",
        "model_inputs": "CSV timestamp and value only; generation settings are simulation truth.",
        "datasets": datasets,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved {len(datasets)} series with {count} observations each to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "synthetic",
        help="Destination directory; existing dataset files are never overwritten.",
    )
    generate(parser.parse_args().output_dir)
