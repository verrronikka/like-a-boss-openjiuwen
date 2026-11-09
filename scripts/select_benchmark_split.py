"""Make a fresh runnable split without exposing reference labels to the agent."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def select_split(source, output, split):
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError("Choose a fresh output directory")
    manifest=json.loads((source/"labels.json").read_text(encoding="utf-8"))
    cases=[c for c in manifest["cases"] if c["split"]==split]
    if not cases:
        raise ValueError("Empty split")
    for case in cases:
        for name in ("tasks","reference"):
            shutil.copytree(source/name/case["task_id"],output/name/case["task_id"])
    digest=hashlib.sha256()
    for path in sorted((output/"tasks").rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(output/"tasks").as_posix().encode()+b"\0"+path.read_bytes())
    selected={**manifest,"cases":cases,"parent_sha256":manifest["dataset_sha256"],"dataset_sha256":digest.hexdigest(),"split":split}
    (output/"labels.json").write_text(json.dumps(selected,ensure_ascii=False,indent=2),encoding="utf-8")
    return selected


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--split",choices=("train","calibration","validation"),required=True)
    args=parser.parse_args()
    selected=select_split(args.source,args.output,args.split)
    print(json.dumps({"count":len(selected["cases"]),"sha256":selected["dataset_sha256"]}))
