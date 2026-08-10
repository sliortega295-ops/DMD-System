#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,os
from pathlib import Path
from typing import Any,Mapping
from distillschedule.runtime.gan_fairness import compare_gan_fairness_receipts

def load(path:Path)->dict[str,Any]:
    value=json.loads(path.read_text())
    if not isinstance(value,dict):raise ValueError(f"expected JSON object: {path}")
    return value
def atomic_json(path:Path,value:Mapping[str,Any])->None:
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_name(f".{path.name}.{os.getpid()}.tmp"); tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n"); os.replace(tmp,path)
def main()->int:
    p=argparse.ArgumentParser(); p.add_argument("--baseline",required=True,type=Path); p.add_argument("--pipeline",required=True,type=Path); p.add_argument("--output",required=True,type=Path); a=p.parse_args()
    try: result=compare_gan_fairness_receipts(load(a.baseline.resolve()),load(a.pipeline.resolve()))
    except (OSError,ValueError,TypeError,KeyError) as exc:
        result={"schema_version":"distillschedule.gan_fairness_pair.v1","status":"FAIL","claimable":False,"error":str(exc)}; atomic_json(a.output.resolve(),result); print(json.dumps(result,sort_keys=True)); return 2
    result=dict(result); result["claimable"]=True; atomic_json(a.output.resolve(),result); print(json.dumps(result,sort_keys=True)); return 0
if __name__=="__main__":raise SystemExit(main())
