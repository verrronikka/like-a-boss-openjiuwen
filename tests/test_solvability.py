import json
import tempfile
import unittest
from pathlib import Path
from scripts.build_solvability_benchmark import build, new_pairs
from scripts.score_solvability import evaluate


class SolvabilityBenchmarkTests(unittest.TestCase):
    def test_labels_are_separate_and_pairs_have_controls(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "benchmark"
            manifest = build(root)
            self.assertEqual(len(manifest["cases"]), 16)
            self.assertEqual(sum(not c["solvable"] for c in manifest["cases"]), 4)
            for file in (root / "tasks").glob("*/task.json"):
                task = json.loads(file.read_text())
                self.assertEqual(set(task), {"task_id", "task_type", "instruction", "reward", "check"})
                self.assertNotIn("expected", file.read_text())
            pairs = new_pairs()
            for name in {c["pair"] for c in pairs}:
                self.assertEqual(sorted(c["solvable"] for c in pairs if c["pair"] == name), [False, True])
            second = build(Path(folder) / "second")
            self.assertEqual(manifest["dataset_sha256"], second["dataset_sha256"])
            with self.assertRaises(FileExistsError):
                build(root)

    def test_evaluator_distinguishes_skip_from_missing_decision_and_false_answer(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "benchmark"
            manifest = build(root)
            positives = [c for c in manifest["cases"] if c["solvable"]]
            negatives = [c for c in manifest["cases"] if not c["solvable"]]
            attempts = [dict(task_id=negatives[0]["task_id"], status="skipped", spent=123),
                        dict(task_id=negatives[1]["task_id"], status="budget_limit", spent=300),
                        dict(task_id=negatives[2]["task_id"], status="submitted", spent=50),
                        dict(task_id=positives[0]["task_id"], status="skipped", spent=10)]
            (root / "tasks" / negatives[2]["task_id"] / "answer.txt").write_text("invented")
            (root / "tasks" / "agent-report.json").write_text(json.dumps(
                dict(model="test", spent=483, results=attempts)))
            report = evaluate(root)
            self.assertEqual(report["correct_skips"], 1)
            self.assertEqual(report["false_answers"], 1)
            self.assertEqual(report["impossible_without_decision"], 2)
            self.assertEqual(report["false_skips"], 1)
            self.assertEqual(report["spent_on_impossible"], 473)
            self.assertEqual(report["solvable_passed"], 0)


if __name__ == "__main__":
    unittest.main()
