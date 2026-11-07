import asyncio
import json
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from harness import Budget, Task, TaskTools, atomic_text, load_tasks, run_tasks, solve
from run_agent import main, resolve_budget
from scripts.score_examples import score_examples


class HarnessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.directory = self.root / "math_1"
        (self.directory / "workspace").mkdir(parents=True)
        self.task = Task("math_1", "math", "Compute 6*7", 5, self.directory)

    def tearDown(self):
        self.temp.cleanup()

    def test_input_does_not_use_expected_and_plan_uses_reward(self):
        for name, reward, expected in [("a", 1, "bait-a"), ("b", 10, "bait-b")]:
            directory = self.root / name
            directory.mkdir()
            (directory / "task.json").write_text(json.dumps({"task_id": name, "task_type": "math",
                "instruction": "Compute 2+2", "reward": reward, "expected": expected}), encoding="utf-8")
        tasks = load_tasks(self.root)
        self.assertEqual([task.task_id for task in tasks], ["b", "a"])
        self.assertNotIn("bait", repr(tasks))

    def test_file_boundary_and_immediate_submission(self):
        tools = TaskTools(self.task)
        for path in ("../task.json", "../../outside.txt", "task.json"):
            with self.assertRaises(ValueError):
                tools.resolve(path)
        with self.assertRaises(ValueError):
            tools.execute("write_file", {"path": "test_mylib.py", "content": "wrong"})
        tools.execute("submit_answer", {"answer": "42"})
        self.assertEqual((self.directory / "answer.txt").read_text().strip(), "42")
        self.assertFalse((self.task.workspace / "answer.txt").exists())

    def test_meeting_budget_comes_from_environment_and_cli_only_lowers_it(self):
        self.assertEqual(resolve_budget(environ={}), 100000)
        self.assertEqual(resolve_budget(environ={"BUDGET": "12345"}), 12345)
        self.assertEqual(resolve_budget(500, {"BUDGET": "12345"}), 500)
        self.assertEqual(resolve_budget(200000, {"BUDGET": "12345"}), 12345)
        for raw in ("bad", "0", "-10", "1.5", ""):
            with self.assertRaises(ValueError):
                resolve_budget(environ={"BUDGET": raw})

    def test_skipping_creates_no_answer_and_repeat_run_is_rejected(self):
        tools = TaskTools(self.task)
        tools.execute("skip_task", {"reason": "Necessary input is missing"})
        self.assertFalse((self.directory / "answer.txt").exists())
        self.assertFalse((self.task.workspace / "answer.txt").exists())
        atomic_text(self.directory / "answer.txt", "old answer")
        with patch("run_agent.load_tasks", return_value=[self.task]):
            self.assertEqual(main(["--tasks-dir", str(self.root)]), 1)
        self.assertEqual((self.directory / "answer.txt").read_text(), "old answer")

    def test_local_score_uses_references_and_awards_nothing_for_skip(self):
        reference = self.root / "references"
        solutions = self.root / "solutions"
        for name, reward in (("a", 3), ("b", 7)):
            folder = reference / name
            folder.mkdir(parents=True)
            atomic_text(folder / "task.json", json.dumps({"task_id": name, "reward": reward,
                        "expected": "42", "check": "exact_int"}))
        atomic_text(solutions / "a" / "answer.txt", "42")
        report = score_examples(reference, solutions)
        self.assertEqual((report["score"], report["max_score"], report["passed"]), (3, 10, 1))
        self.assertEqual(report["tasks"][1]["status"], "no_answer")

    def test_local_coding_score_checks_program_and_rejects_changed_tests(self):
        reference = self.root / "references" / "coding_1"
        solution = self.root / "solutions" / "coding_1"
        atomic_text(reference / "task.json", json.dumps({"task_id": "coding_1", "reward": 2,
                    "expected": "tests_pass", "check": "unittest"}))
        tests = "import unittest\nfrom mylib import add\nclass Tests(unittest.TestCase):\n def test_add(self): self.assertEqual(add(2,3),5)\n"
        atomic_text(reference / "workspace" / "test_mylib.py", tests)
        atomic_text(solution / "workspace" / "test_mylib.py", tests)
        atomic_text(solution / "workspace" / "mylib.py", "def add(a,b): return a-b\n")
        atomic_text(solution / "answer.txt", "DONE")
        self.assertEqual(score_examples(reference.parent, solution.parent)["score"], 0)
        atomic_text(solution / "workspace" / "mylib.py", "def add(a,b): return a+b\n")
        self.assertEqual(score_examples(reference.parent, solution.parent)["score"], 2)
        atomic_text(solution / "workspace" / "test_mylib.py", "# no tests\n")
        self.assertEqual(score_examples(reference.parent, solution.parent)["score"], 0)

    def test_budget_counts_full_input_and_does_not_double_count_reasoning(self):
        budget = Budget(100)
        budget.add(SimpleNamespace(input_tokens=100, output_tokens=30, reasoning_tokens=20))
        self.assertEqual(budget.spent, 40)
        self.assertEqual(budget.remaining, 60)
        with self.assertRaises(ValueError):
            budget.add(None)

    def test_atomic_replace(self):
        path = self.root / "report.json"
        atomic_text(path, "first")
        atomic_text(path, "second")
        self.assertEqual(path.read_text(), "second")
        self.assertEqual(list(self.root.glob(".agent-*")), [])

    def test_python_runs_in_workspace_without_api_key(self):
        tools = TaskTools(self.task)
        with patch.dict("os.environ", {"LLM_API_KEY": "never-pass-this-to-python"}):
            output = tools.execute("run_python", {"code": "import os; print(os.getenv('LLM_API_KEY')); print(6*7)"})
        data = json.loads(output)
        self.assertEqual(data["exit_code"], 0)
        self.assertEqual(data["stdout"].splitlines(), ["None", "42"])

    def test_insufficient_budget_makes_no_model_call(self):
        class NoCalls:
            async def invoke(self, *args, **kwargs):
                raise AssertionError("No call expected")
        args = SimpleNamespace(max_steps=3, task_budget=1000, max_output=128)
        result = asyncio.run(solve(self.task, NoCalls(), Budget(1), args, time.monotonic() + 10))
        self.assertEqual(result["status"], "budget_limit")
        self.assertFalse((self.directory / "answer.txt").exists())

    def test_real_openjiuwen_model_against_local_test_api(self):
        requests = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                requests.append(body)
                if len(requests) == 1:
                    name, arguments = "run_python", {"code": "print(6*7)"}
                else:
                    name, arguments = "submit_answer", {"answer": "42"}
                payload = {"id": "test-response", "object": "chat.completion", "created": 1,
                    "model": body["model"], "choices": [{"index": 0, "finish_reason": "tool_calls",
                    "message": {"role": "assistant", "content": "", "tool_calls": [{"id": f"call-{len(requests)}",
                    "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}]}}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120,
                              "completion_tokens_details": {"reasoning_tokens": 5}}}
                data = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        args = SimpleNamespace(model="qwen3.8-27b-noreason", budget=1000, task_budget=500,
                               max_output=128, max_steps=4, time_limit=30, report=self.root / "report.json")
        try:
            with patch.dict("os.environ", {"LLM_BASE_URL": f"http://127.0.0.1:{server.server_port}/v1",
                                           "LLM_API_KEY": "test-key-not-a-real-credential",
                                           "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost"}):
                code = asyncio.run(run_tasks([self.task], args))
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
        self.assertEqual(code, 0, f"requests={len(requests)}; {args.report.read_text(encoding='utf-8')}")
        self.assertEqual(len(requests), 2)
        self.assertEqual((self.directory / "answer.txt").read_text().strip(), "42")
        report = json.loads(args.report.read_text(encoding="utf-8"))
        self.assertEqual(report["spent"], 60)
        self.assertEqual(report["results"][0]["status"], "submitted")
        self.assertEqual(requests[1]["messages"][-1]["role"], "tool")
        self.assertIn("Remaining overall", requests[1]["messages"][1]["content"])


if __name__ == "__main__":
    unittest.main()
