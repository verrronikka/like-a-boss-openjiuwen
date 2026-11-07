"""Command line entry point for the like_a_boss prototype."""
import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from harness import ALLOWED_MODELS, load_tasks, run_tasks


def main():
    parser = argparse.ArgumentParser(description="Budget-aware openJiuwen agent prototype")
    parser.add_argument("--tasks-dir", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true", help="List the plan without calling a model or changing files")
    parser.add_argument("--model", choices=ALLOWED_MODELS, default="qwen3.8-27b-noreason")
    parser.add_argument("--budget", type=int, default=20000, help="Local total budget in output-equivalent tokens; not an official competition value")
    parser.add_argument("--task-budget", type=int, default=4000)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--max-output", type=int, default=1024)
    parser.add_argument("--limit", type=int, help="Only attempt the first N tasks in the plan")
    parser.add_argument("--time-limit", type=int, default=3300, help="Total wall time in seconds")
    parser.add_argument("--report", type=Path, help="Default: tasks-dir/agent-report.json")
    args = parser.parse_args()
    for name in ("budget", "task_budget", "max_steps", "max_output", "time_limit"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    try:
        tasks = load_tasks(args.tasks_dir)
        if not tasks:
            parser.error("No task.json files found under --tasks-dir")
        if args.limit:
            tasks = tasks[:args.limit]
        if args.dry_run:
            print(json.dumps([task.summary() for task in tasks], ensure_ascii=False, indent=2))
            return 0
        for name in ("LLM_BASE_URL", "LLM_API_KEY"):
            if not os.environ.get(name):
                parser.error(f"Set {name} before a real run, or use --dry-run")
        return asyncio.run(run_tasks(tasks, args))
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
