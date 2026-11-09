import asyncio
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from harness import Budget, Task, TaskTools, check_authored_python, configure_sdk, resource_snapshot, solve


class FeasibilityTests(unittest.TestCase):
    def test_snapshot_excludes_labels_and_marks_truncation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "task.json").write_text('{"expected":"secret label"}')
            (root / "facts.md").write_text("a" * 100)
            task = Task("x", "search", "Find a fact", 1, root)
            snapshot = resource_snapshot(task, max_bytes=20)
            self.assertFalse(snapshot["complete"])
            self.assertNotIn("secret label", str(snapshot))
            self.assertEqual(len(snapshot["files"]["facts.md"]), 20)

    def test_obvious_caller_workarounds_are_rejected_without_blocking_test_runner(self):
        for code in ("import sys; sys._getframe()", "import inspect; inspect.stack()"):
            with self.assertRaises(ValueError):
                check_authored_python(code)
        check_authored_python("import runpy; runpy.run_path('test_mylib.py', run_name='__main__')")
        with tempfile.TemporaryDirectory() as directory:
            task = Task("x", "coding", "Implement a pure function", 1, Path(directory))
            tools = TaskTools(task)
            with self.assertRaises(ValueError):
                tools.execute("write_file", {"path": "mylib.py", "content": "import sys; sys._getframe()"})
            self.assertFalse((Path(directory) / "mylib.py").exists())

    def test_review_skip_is_charged_and_creates_no_answer(self):
        configure_sdk()
        from openjiuwen.core.foundation.llm import AssistantMessage, UsageMetadata
        class Reviewer:
            async def invoke(self, messages, **kwargs):
                self.calls = getattr(self, "calls", 0) + 1
                return AssistantMessage(content='{"decision":"skip","reason":"Two mandatory constraints contradict"}',
                    usage_metadata=UsageMetadata(input_tokens=100, output_tokens=20))
        with tempfile.TemporaryDirectory() as directory:
            task = Task("x", "math", "Return an integer larger than 8 and smaller than 3", 1, Path(directory))
            args = SimpleNamespace(solvability_checks=True, max_steps=4, task_budget=1000, max_output=128)
            budget, model = Budget(1000), Reviewer()
            result = asyncio.run(solve(task, model, budget, args, time.monotonic() + 10))
            self.assertEqual(result["status"], "skipped")
            self.assertEqual(budget.spent, 60)
            self.assertEqual(model.calls, 2)
            self.assertFalse((Path(directory) / "answer.txt").exists())

    def test_inconclusive_review_cannot_be_bypassed_by_submission(self):
        configure_sdk()
        from openjiuwen.core.foundation.llm import AssistantMessage, UsageMetadata
        from openjiuwen.core.foundation.llm.schema.tool_call import ToolCall
        class Model:
            calls = 0
            async def invoke(self, messages, **kwargs):
                self.calls += 1
                usage = UsageMetadata(input_tokens=100, output_tokens=20)
                if self.calls == 1:
                    return AssistantMessage(content="not JSON", usage_metadata=usage)
                if self.calls == 2:
                    return AssistantMessage(content="", usage_metadata=usage, tool_calls=[ToolCall(
                        id="guess", type="function", name="submit_answer", arguments='{"answer":"45"}')])
                return AssistantMessage(content='{"decision":"skip","reason":"Required initial value is absent"}',
                                        usage_metadata=usage)
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(solvability_checks=True, max_steps=4, task_budget=1000, max_output=128)
            task = Task("x", "math", "Initial value is unknown; return the exact final value", 1, Path(directory))
            budget, model = Budget(1000), Model()
            result = asyncio.run(solve(task, model, budget, args, time.monotonic() + 10))
            self.assertEqual(result["status"], "skipped")
            self.assertEqual(model.calls, 4)
            self.assertEqual(budget.spent, 120)
            self.assertFalse((Path(directory) / "answer.txt").exists())

    def test_skip_audit_can_reject_a_false_refusal(self):
        configure_sdk()
        from openjiuwen.core.foundation.llm import AssistantMessage, UsageMetadata
        from openjiuwen.core.foundation.llm.schema.tool_call import ToolCall
        class Model:
            calls = 0
            async def invoke(self, messages, **kwargs):
                self.calls += 1
                usage = UsageMetadata(input_tokens=100, output_tokens=20)
                if self.calls == 1:
                    return AssistantMessage(content='{"reason":"An irrelevant fact is absent","decision":"skip"}', usage_metadata=usage)
                if self.calls == 2:
                    self.audit_prompt = messages[0].content
                    return AssistantMessage(content='{"reason":"The requested value is fully determined","decision":"proceed"}', usage_metadata=usage)
                return AssistantMessage(content="", usage_metadata=usage, tool_calls=[ToolCall(
                    id="answer", type="function", name="submit_answer", arguments='{"answer":"42"}')])
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(solvability_checks=True, max_steps=4, task_budget=1000, max_output=128)
            task = Task("x", "math", "Compute 6*7", 1, Path(directory))
            budget, model = Budget(1000), Model()
            result = asyncio.run(solve(task, model, budget, args, time.monotonic() + 10))
            self.assertEqual(result["status"], "submitted")
            self.assertIn("Independently audit", model.audit_prompt)
            self.assertEqual(model.calls, 3)
            self.assertEqual(budget.spent, 90)

    def test_actor_refusal_also_requires_confirmation(self):
        configure_sdk()
        from openjiuwen.core.foundation.llm import AssistantMessage, UsageMetadata
        from openjiuwen.core.foundation.llm.schema.tool_call import ToolCall
        class Model:
            calls = 0
            async def invoke(self, messages, **kwargs):
                self.calls += 1
                usage = UsageMetadata(input_tokens=100, output_tokens=20)
                if self.calls in (1, 3):
                    return AssistantMessage(content='{"reason":"All required facts are provided","decision":"proceed"}', usage_metadata=usage)
                call = ToolCall(id=str(self.calls), type="function", name="skip_task" if self.calls == 2 else "submit_answer",
                    arguments='{"reason":"unnecessary fact absent"}' if self.calls == 2 else '{"answer":"42"}')
                return AssistantMessage(content="", usage_metadata=usage, tool_calls=[call])
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(solvability_checks=True, max_steps=4, task_budget=1000, max_output=128)
            task = Task("x", "math", "Compute 6*7", 1, Path(directory))
            budget, model = Budget(1000), Model()
            result = asyncio.run(solve(task, model, budget, args, time.monotonic() + 10))
            self.assertEqual(result["status"], "submitted")
            self.assertEqual(model.calls, 4)
            self.assertEqual(budget.spent, 120)


if __name__ == "__main__":
    unittest.main()
