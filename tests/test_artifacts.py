"""Shared artifact writers keep saved bytes; shared names have one definition."""

import csv
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from model_pipelining.artifacts import write_csv, write_json
from model_pipelining.evidence import PROVIDER_WORKFLOWS, EvidenceMetadata
from model_pipelining.evidence_workflows import REQUIRED_WORKFLOW_CHECKS, WORKFLOW_MANIFESTS
from model_pipelining.experiments import EVIDENCE_WORKFLOWS, EvaluationPlan, EvidencePlan, read_observations
from model_pipelining.models import BENCHMARK_MODELS, LastValue, Observation, TrainingMean


class WriterTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.root = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def test_json_bytes_match_the_previous_layout_and_creation_is_exclusive(self):
        value = {"b": [1, 2.5, None], "a": "café"}
        path = self.root / "value.json"
        write_json(path, value)
        self.assertEqual(path.read_bytes(), (json.dumps(value, indent=2, allow_nan=False) + "\n").encode())
        with self.assertRaises(FileExistsError):
            write_json(path, {})
        write_json(path, {"status": "completed"}, overwrite=True)
        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"status": "completed"})

    def test_nonfinite_json_is_rejected_before_a_file_is_created(self):
        path = self.root / "bad.json"
        with self.assertRaises(ValueError):
            write_json(path, {"value": float("nan")})
        self.assertFalse(path.exists())

    def test_csv_bytes_match_the_previous_dictwriter_layout(self):
        rows = [{"time": "2024-01-01", "value": 1.5}, {"time": "2024-01-02", "value": None}]
        expected = io.StringIO(newline="")
        writer = csv.DictWriter(expected, fieldnames=["time", "value"])
        writer.writeheader()
        writer.writerows(rows)
        for name, fields in (("dict.csv", rows[0]), ("tuple.csv", ("time", "value"))):
            path = self.root / name
            write_csv(path, fields, rows)
            self.assertEqual(path.read_bytes(), expected.getvalue().encode())
            with self.assertRaises(FileExistsError):
                write_csv(path, fields, rows)


class SharedNameTests(unittest.TestCase):
    def test_benchmark_names_come_from_point_models(self):
        self.assertEqual(BENCHMARK_MODELS, {"training-mean": TrainingMean, "last-value": LastValue})
        with self.assertRaisesRegex(ValueError, "unique training-mean/last-value names"):
            EvaluationPlan(benchmarks=("persistence",))

    def test_workflow_tables_cover_exactly_the_experiment_workflows(self):
        self.assertEqual(EVIDENCE_WORKFLOWS, ("predictive_checks", "prior_influence", "sequential_checks"))
        self.assertEqual(tuple(WORKFLOW_MANIFESTS), EVIDENCE_WORKFLOWS)
        self.assertEqual(WORKFLOW_MANIFESTS["prior_influence"], "prior_influence.json")
        self.assertEqual(set(REQUIRED_WORKFLOW_CHECKS), set(EVIDENCE_WORKFLOWS))
        self.assertEqual(PROVIDER_WORKFLOWS, EVIDENCE_WORKFLOWS + ("calibration",))
        self.assertEqual(EvidenceMetadata.__dataclass_fields__["workflows"].default, PROVIDER_WORKFLOWS)
        with self.assertRaisesRegex(ValueError, "predictive_checks, prior_influence or sequential_checks"):
            EvidencePlan(required_workflows=("calibration",))

    def test_read_observations_returns_immediate_daily_values(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "series.csv"
            path.write_text("timestamp,value\n2024-01-01T00:00:00+00:00,1\n2024-01-02T00:00:00+00:00,2.5\n",
                            encoding="utf-8")
            observations = read_observations(path)
        self.assertEqual([item.value for item in observations], [1.0, 2.5])
        self.assertTrue(all(type(item) is Observation for item in observations))


if __name__ == "__main__":
    unittest.main()
