import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from scripts.build_large_benchmark import build_large
from scripts.select_benchmark_split import select_split


class LargeBenchmarkTests(unittest.TestCase):
    def test_size_split_leakage_and_coding_reference_contracts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"large"
            manifest=build_large(root)
            cases=manifest["cases"]
            self.assertEqual(len(cases),200)
            self.assertEqual(sum(not c["solvable"] for c in cases),40)
            self.assertEqual(Counter(c["split"] for c in cases),dict(train=120,calibration=40,validation=40))
            families={}
            for c in cases:
                self.assertEqual(families.setdefault(c["family"],c["split"]),c["split"])
                metadata=json.loads((root/"tasks"/c["task_id"]/"task.json").read_text())
                self.assertEqual(set(metadata),{"task_id","task_type","instruction","reward","check"})
                self.assertFalse((root/"tasks"/c["task_id"]/"oracle.py").exists())
                if c["task_type"]=="coding":
                    ref=root/"reference"/c["task_id"]
                    shutil.copyfile(ref/"oracle.py",ref/"workspace"/"mylib.py")
                    tested=subprocess.run([sys.executable,"test_mylib.py"],cwd=ref/"workspace",capture_output=True,timeout=5)
                    self.assertEqual(tested.returncode==0,c["solvable"],tested.stderr.decode(errors="replace"))
            self.assertEqual(len(families),20)
            selected=select_split(root,Path(tmp)/"validation","validation")
            self.assertEqual(len(selected["cases"]),40)
            self.assertEqual(sum(not c["solvable"] for c in selected["cases"]),8)
            self.assertFalse((Path(tmp)/"validation"/"tasks"/"labels.json").exists())
