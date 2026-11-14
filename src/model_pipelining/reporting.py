"""Draw a chart from saved evaluation results; never fit or call a model."""

import csv
from datetime import datetime
import json
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


def plot_run(run_directory: str | Path) -> Path:
    """Save forecast.png from an existing run, refusing to overwrite a chart.

    Failed predictions appear as gaps. The title includes run status and scored
    coverage so a partial run is not presented as a complete success.
    """
    run = Path(run_directory).expanduser()
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] not in ("completed", "failed"):
        raise ValueError("Reporting requires a finished run")
    scores = json.loads((run / "scores.json").read_text(encoding="utf-8"))
    with (run / "evaluation.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    figure = Figure(figsize=(10, 4), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots()
    times = [datetime.fromisoformat(row["target_time"]) for row in rows]
    axes.plot(times, [float(row["actual"]) for row in rows], label="Actual", linewidth=1)
    axes.plot(times, [float(row["prediction"]) if row["prediction"] else float("nan")
                      for row in rows], label="Forecast", linewidth=1.5)
    axes.set(title=f"{manifest['model']['name']} — {manifest['status']} "
                   f"({scores['scored_count']}/{scores['expected_count']} scored)",
             xlabel="Target date (UTC)", ylabel="Value (input units)")
    axes.legend()
    figure.autofmt_xdate()
    destination = run / "forecast.png"
    with destination.open("xb") as stream:
        figure.savefig(stream, format="png", dpi=150)
    return destination
