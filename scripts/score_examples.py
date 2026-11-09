"""Local sample checker only; never imported by the solving agent."""
import argparse
import json
import os
import subprocess
import sys
from fractions import Fraction
from pathlib import Path


def score_examples(reference_dir, tasks_dir, task_ids=None):
    results = []
    for metadata in sorted(Path(reference_dir).resolve().glob("*/task.json")):
        if task_ids is not None and metadata.parent.name not in task_ids:
            continue
        task = json.loads(metadata.read_text(encoding="utf-8"))
        if "expected" not in task:
            raise ValueError("Local scoring requires the sample expected field; hidden tasks cannot be scored here")
        directory = Path(tasks_dir).resolve() / metadata.parent.name
        answer = directory / "answer.txt"
        result = {"task_id": task["task_id"], "reward": task["reward"], "passed": False}
        if not answer.is_file():
            result["status"] = "no_answer"
        else:
            actual = answer.read_text(encoding="utf-8").strip()
            kind = task["check"]
            if kind == "exact_str":
                result["passed"] = actual == str(task["expected"])
            elif kind == "exact_int":
                try:
                    result["passed"] = int(actual) == int(task["expected"])
                except ValueError:
                    pass
            elif kind == "exact_fraction":
                try:
                    result["passed"] = Fraction(actual) == Fraction(task["expected"])
                except (ValueError, ZeroDivisionError):
                    pass
            elif kind == "unittest":
                source = metadata.parent / "workspace"
                workspace = directory / "workspace"
                tests = list(source.rglob("test*.py"))
                unchanged = bool(tests) and all(
                    (workspace / test.relative_to(source)).is_file()
                    and (workspace / test.relative_to(source)).read_bytes() == test.read_bytes()
                    for test in tests)
                if actual == "DONE" and unchanged:
                    env = {key: os.environ[key] for key in ("SYSTEMROOT", "WINDIR", "PATH", "TEMP", "TMP") if key in os.environ}
                    env["PYTHONDONTWRITEBYTECODE"] = "1"
                    try:
                        checked = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "."],
                            cwd=workspace, env=env, capture_output=True, timeout=15)
                        result["passed"] = checked.returncode == 0
                    except subprocess.TimeoutExpired:
                        result["status"] = "test_timeout"
                elif not unchanged:
                    result["status"] = "tests_changed_or_missing"
            else:
                raise ValueError(f"Unsupported check type: {kind}")
            result.setdefault("status", "passed" if result["passed"] else "wrong_answer")
        results.append(result)
    if not results:
        raise ValueError("No sample tasks found in reference directory")
    return {"scope": "local_samples_only", "score": sum(r["reward"] for r in results if r["passed"]),
            "max_score": sum(r["reward"] for r in results),
            "passed": sum(r["passed"] for r in results), "total": len(results), "tasks": results}


def main():
    parser = argparse.ArgumentParser(description="Score sample answers locally, using original sample references")
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--tasks-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(score_examples(args.reference_dir, args.tasks_dir), ensure_ascii=False, indent=2))
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
