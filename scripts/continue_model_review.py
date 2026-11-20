"""Continue prepared model-review work, preserving compatible completed artifacts.

Independent model/scenario jobs use separate processes and output directories.
Stop the original study process before invoking this command.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime, timedelta
import hashlib
from pathlib import Path
import uuid

from model_pipelining.experiment_review import build_experiment_review
from model_pipelining.evidence_workflows import run_experiment_evidence, plot_experiment_evidence
from model_pipelining.experiments import ExperimentSpec
from model_pipelining.backtest import run_backtest
from model_pipelining.reporting import compare_runs, plot_run
from review_model_study import (ROOT, MODEL_CONFIGS, read_json, write_json, case_metrics,
                                summarize, plot_summary)


def completed(path, filename):
    """Reuse only complete artifacts made with the current native sources."""
    if not path.exists():
        return False
    manifest = read_json(path / filename)
    if manifest.get("status") != "completed":
        if manifest.get("status") == "failed":
            raise ValueError(f"Investigate the retained failed artifact before continuing: {path}")
        preserved = path.with_name(f"{path.name}-interrupted-{uuid.uuid4().hex[:8]}")
        path.rename(preserved)
        print(f"Preserved partial artifact: {preserved}", flush=True)
        return False
    for name, digest in manifest["source_sha256"].items():
        if hashlib.sha256((ROOT / "src/model_pipelining" / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Completed artifact has incompatible source: {path}/{name}")
    return True


def run_job(study, scenario, name, plan=None):
    root = Path(study)
    plan = read_json(root / "study.json") if plan is None else plan
    spec = ExperimentSpec.from_dict(plan["experiment"])
    settings = plan["models"][name]
    kind = MODEL_CONFIGS[name][0]
    model = kind.from_configuration(settings["reference"])
    alternatives = {prior: kind.from_configuration(value) for prior, value in settings["alternatives"].items()}
    base = root / scenario / name
    common = dict(dataset=scenario, snapshot_id=plan["snapshots"][scenario], store=root / "store", experiment=spec)
    for prior, selected in (("reference", model), *alternatives.items()):
        path = base / prior
        if completed(path, "manifest.json"):
            manifest = read_json(path / "manifest.json")
            expected = settings["reference"] if prior == "reference" else settings["alternatives"][prior]
            if (manifest["model"]["configuration"] != expected
                    or manifest["input"]["snapshot_id"] != plan["snapshots"][scenario]
                    or manifest["input"]["dataset"] != scenario or manifest["experiment"] != spec.to_dict()
                    or manifest["evaluation"]["interval_levels"] != plan["interval_levels"]):
                raise ValueError(f"Completed forecast differs from saved study design: {path}")
            case_metrics(path)
            print(f"Reused forecast: {scenario}/{name}/{prior}", flush=True)
        else:
            print(f"Forecast: {scenario}/{name}/{prior}", flush=True)
            run_backtest(**common, model=selected, interval_levels=tuple(plan["interval_levels"]), output_directory=path)
        if prior == "reference" and not (path / "forecast.png").exists():
            plot_run(path)
    evidence = base / "evidence"
    if not completed(evidence, "evidence.json"):
        print(f"Evidence: {scenario}/{name}", flush=True)
        run_experiment_evidence(**common, model=model, scenarios=alternatives,
            issued_at=spec.evaluation.final_start - timedelta(seconds=1), output_directory=evidence)
        plot_experiment_evidence(evidence)
    # Reassemble even reused sources: native review checks child identities,
    # completion receipts, versions, configuration and current source inventories.
    review = base / "review"
    if review.exists():
        review = base / f"review-continuation-{uuid.uuid4().hex[:8]}"
    build_experiment_review(forecast_directory=base / "reference", evidence_directory=evidence,
        calibration_directories=(root / "calibration" / name,), require_calibration=True, output_directory=review)
    status = read_json(review / "report.json")["status"]
    return {"scenario": scenario, "model": name, "review_status": status,
            "latest_review": str(review.relative_to(root))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-directory", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.workers <= 9:
        raise ValueError("Choose 1 to 9 independent workers")
    root = args.study_directory.expanduser().absolute()
    plan = read_json(root / "study.json")
    if plan.get("kind") != "model_review_study" or plan.get("status") == "completed":
        raise ValueError("Use a prepared, unfinished model-review study")
    for name in plan["models"]:
        path = root / "calibration" / name
        if not completed(path, "calibration.json"):
            raise ValueError("Complete compatible calibration before continuing")
        if read_json(path / "calibration.json")["model"]["configuration"] != plan["models"][name]["reference"]:
            raise ValueError("Calibration differs from saved study design")
    script_bytes = Path(__file__).read_bytes()
    script_hash = hashlib.sha256(script_bytes).hexdigest()
    sources = root / "continuation-sources"
    sources.mkdir(exist_ok=True)
    (sources / f"{script_hash}.py").write_bytes(script_bytes)
    plan.setdefault("continuations", []).append({"started_at": datetime.now(UTC).isoformat(),
        "script_sha256": script_hash, "workers": args.workers,
        "scope": "Reuse current-source complete artifacts; retain partial artifacts before recomputing"})
    plan.update(status="running", results=[], job_errors=[])
    write_json(root / "study.json", plan)
    design = {key: plan[key] for key in ("experiment", "models", "snapshots", "interval_levels")}
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_job, str(root), scenario, name, design): (scenario, name)
                   for scenario in plan["scenario_order"] for name in plan["models"]}
        for future in as_completed(futures):
            try:
                plan["results"].append(future.result())
            except Exception as error:
                scenario, name = futures[future]
                plan["job_errors"].append({"scenario": scenario, "model": name,
                                           "error": f"{type(error).__name__}: {error}"})
            write_json(root / "study.json", plan)
    if plan["job_errors"]:
        plan.update(status="failed", finished_at=datetime.now(UTC).isoformat())
        write_json(root / "study.json", plan)
        return 1
    for scenario in plan["scenario_order"]:
        for first, second in (("constant", "learned-noise"), ("constant", "joint"), ("learned-noise", "joint")):
            compare_runs(root / scenario / first / "reference", root / scenario / second / "reference",
                output_directory=root / scenario / "comparisons" / f"{first}-vs-{second}")
    summary = summarize(root)
    plot_summary(root, summary)
    plan.update(status="completed" if all(row["review_status"] == "ready_for_review" for row in plan["results"]) else "failed",
                finished_at=datetime.now(UTC).isoformat())
    write_json(root / "study.json", plan)
    return 0 if plan["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
