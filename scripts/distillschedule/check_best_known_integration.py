#!/usr/bin/env python3
"""Final-HEAD CPU acceptance; never invokes GPU, SSH, or a model worker."""
from __future__ import annotations
import ast
import compileall
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"src"),str(ROOT)]
from distillschedule.workloads.best_known.registry import CANONICAL_IDS, load_profile, source_layout


def main() -> int:
    for folder in ("src","runtime","scripts/distillschedule"):
        if not compileall.compile_dir(ROOT/folder,quiet=1):raise SystemExit(f"compile failed: {folder}")
    for p in (ROOT/"scripts").rglob("*.sh"):
        subprocess.run(["bash","-n",str(p)],check=True)
    manifest=json.loads((ROOT/"docs/workloads/source_import_map.json").read_text())
    for r in manifest["files"]:
        p=ROOT/r["target"]
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=r["integrated_sha256"]:
            raise SystemExit(f"integrated source differs from reviewed import map: {r['target']}")
    # Verify local-package AST edges without importing Torch/FA2/FA3 or models.
    missing=[]
    for row in manifest["files"]:
        p=ROOT/row["target"]
        if p.suffix!=".py" or row["target"].startswith("tests/"):continue
        for n in ast.walk(ast.parse(p.read_text())):
            if isinstance(n,ast.ImportFrom) and not n.level and n.module:
                name=n.module
                if not name.startswith(("runtime.","distillschedule.")):continue
                rel=Path(*name.split("."));base=ROOT/("src" if name.startswith("distillschedule.") else "")/rel
                if not base.with_suffix(".py").is_file() and not (base/"__init__.py").is_file():
                    missing.append((row["target"],name))
    # Native workers also use spec_from_file_location(ROOT / "...py"). These
    # edges are invisible to import-statement-only checks and caused a real
    # missing static Full5 worker on the integrated Wan/Helios path.
    for p in (ROOT/"scripts/distillschedule").glob("probe_*.py"):
        for n in ast.walk(ast.parse(p.read_text())):
            if (isinstance(n, ast.BinOp) and isinstance(n.op, ast.Div)
                    and isinstance(n.left, ast.Name) and n.left.id == "ROOT"
                    and isinstance(n.right, ast.Constant) and isinstance(n.right.value, str)
                    and n.right.value.endswith(".py") and not (ROOT/n.right.value).is_file()):
                missing.append((str(p.relative_to(ROOT)), n.right.value))
    if missing:raise SystemExit(f"missing local source imports: {missing}")
    from distillschedule.runtime.nsys_range import StaticFSDP2Full5NsightRange
    assert not StaticFSDP2Full5NsightRange(variant="no-gan", enabled=False, rank=0, local_rank=0).enabled
    for name in sorted(CANONICAL_IDS):source_layout(ROOT,load_profile(name,root=ROOT))
    subprocess.run([sys.executable,str(ROOT/"scripts/distillschedule/verify_helios_compat_source.py")],check=True)
    subprocess.run([sys.executable,"-m","unittest","discover","-s",str(ROOT/"tests"),"-p","test_refactor_best_known*.py","-v"],cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT/"src")+os.pathsep+str(ROOT)}, check=True)
    print(json.dumps({"status":"PASS_CPU_INTEGRATION","canonical_profiles":5,"gpu_server":"NOT_RUN_ON_INTEGRATION_BRANCH"}))
    return 0
if __name__=="__main__":raise SystemExit(main())
