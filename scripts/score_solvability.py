"""Evaluate solvability decisions with labels unavailable to the agent."""
import argparse
import json
from pathlib import Path
try:
    from .score_examples import score_examples
except ImportError:
    from score_examples import score_examples


def evaluate(benchmark_dir):
    root = Path(benchmark_dir)
    manifest = json.loads((root / "labels.json").read_text(encoding="utf-8"))
    report = json.loads((root / "tasks" / "agent-report.json").read_text(encoding="utf-8"))
    attempts = {r["task_id"]: r for r in report["results"]}
    solvable_ids = {c["task_id"] for c in manifest["cases"] if c["solvable"]}
    checked = score_examples(root / "reference", root / "tasks", task_ids=solvable_ids)
    checked_by_id = {r["task_id"]: r for r in checked["tasks"]}
    decisions = []
    for case in manifest["cases"]:
        attempt = attempts.get(case["task_id"], {})
        has_answer = (root / "tasks" / case["task_id"] / "answer.txt").exists()
        explicit_skip = attempt.get("status") == "skipped" and not has_answer
        outcome = checked_by_id[case["task_id"]]["status"] if case["solvable"] else (
            "false_answer" if has_answer else "correct_skip" if explicit_skip else "no_decision")
        decisions.append({**case, "outcome": outcome, "agent_status": attempt.get("status", "unattempted"),
                          "spent": attempt.get("spent", 0), "skip_reason": attempt.get("skip_reason")})
    negative = [d for d in decisions if not d["solvable"]]
    positive = [d for d in decisions if d["solvable"]]
    return {"dataset_sha256": manifest["dataset_sha256"], "model": report["model"],
            "total": len(decisions), "solvable": len(positive), "impossible": len(negative),
            "solvable_passed": checked["passed"], "false_skips": sum(d["agent_status"] == "skipped" for d in positive),
            "correct_skips": sum(d["outcome"] == "correct_skip" for d in negative),
            "false_answers": sum(d["outcome"] == "false_answer" for d in negative),
            "impossible_without_decision": sum(d["outcome"] == "no_decision" for d in negative),
            "score": checked["score"], "max_score": checked["max_score"],
            "spent": report["spent"], "spent_on_impossible": sum(d["spent"] for d in negative), "cases": decisions}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.benchmark_dir), ensure_ascii=False, indent=2))
