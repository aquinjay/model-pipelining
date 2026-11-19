"""Verify the frozen plan and its input guards before evaluation."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from model_pipelining.experiments import EvaluationPlan, ExperimentSpec, ForecastTask
from model_pipelining.snapshots import register_snapshot
from scripts.check_noaa_later_period import evaluate, freeze
from scripts.study_noaa_seasonality import TemperatureRegression


class NoaaLaterPeriodTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        history = self.root / "source.csv"
        self.prefix = b"timestamp,value\n2024-12-31T00:00:00Z,7.5\n"
        history.write_bytes(self.prefix)
        receipt = register_snapshot(history, store=self.root / "snapshots",
                                    dataset="noaa-central-park-tmax")
        experiment = ExperimentSpec(task=ForecastTask(minimum_history=365),
                                   evaluation=EvaluationPlan(benchmarks=("last-value",)))
        self.development = self.root / "development"
        self.development.mkdir()
        (self.development / "design.json").write_text(json.dumps(dict(
            source=str(self.root), dataset=receipt.dataset, snapshot_id=receipt.snapshot_id,
            target_dates=["2024-01-01", "2024-12-31"], experiment=experiment.to_dict(),
            models={
                "annual-plus-lag": TemperatureRegression("annual-plus-lag", True, True).configuration,
                "lag-only": TemperatureRegression("lag-only", False, True).configuration,
            })))
        self.study = self.root / "study"

    def test_freeze_sets_final_date_boundaries_and_preserves_input(self):
        plan_path = freeze(self.development, self.study)
        plan = json.loads(plan_path.read_bytes())
        evaluation = plan["experiment"]["evaluation"]
        self.assertEqual(evaluation["role"], "final")
        self.assertEqual(evaluation["final_start"], "2025-01-01T00:00:00+00:00")
        self.assertEqual(evaluation["final_end"], "2026-01-01T00:00:00+00:00")
        self.assertEqual(plan["cases"], 365)
        self.assertEqual((self.study / "history-2023-2024.csv").read_bytes(), self.prefix)
        self.assertEqual(
            hashlib.sha256(plan_path.read_bytes()).hexdigest(),
            (self.study / "plan.sha256").read_text().strip())

    def test_changes_to_frozen_plan_or_history_fail_before_reading_later_data(self):
        plan = freeze(self.development, self.study)
        original = plan.read_bytes()
        plan.write_bytes(original + b" ")
        with self.assertRaisesRegex(ValueError, "frozen plan changed"):
            evaluate(self.study, self.root / "unread.csv")
        plan.write_bytes(original)
        (self.study / "history-2023-2024.csv").write_bytes(self.prefix + b" ")
        with self.assertRaisesRegex(ValueError, "frozen development inputs changed"):
            evaluate(self.study, self.root / "unread.csv")
