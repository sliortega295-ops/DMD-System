#!/usr/bin/env python3
"""Bind a completed native pair to one audited C29 diagnostic descendant."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from distillschedule.runtime.livewan_two_cycle_profile import SCHEMA,VARIANT,sha256

ALLOWED={
    'src/distillschedule/runtime/livewan_crossscale_worker.py',
    'src/distillschedule/runtime/livewan_two_cycle_profile.py',
    'tools/assets/build_livewan_two_cycle_profile_gate.py',
    'scripts/distillschedule/run_livewan_crossscale_world16.py',
    'scripts/distillschedule/launch_livewan_crossscale_world16_node.sh',
    'scripts/distillschedule/summarize_livewan_continuous_world16.py',
    'tests/test_refactor_livewan_two_cycle_profile.py',
    'tests/test_refactor_livewan_crossscale_profile.py',
}
WORKER='src/distillschedule/runtime/livewan_crossscale_worker.py'
CHANGED_FUNCTIONS={'_continuous_pipeline_contract','run_worker','_run_pipeline_8x8_continuous'}

def verify_worker_ast(before: str, after: str) -> dict:
    def definitions(source):
        return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(source).body
                if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))}
    a,b=definitions(before),definitions(after)
    if a.keys()!=b.keys(): raise ValueError('worker function/class set changed')
    changed={k for k in a if a[k]!=b[k]}
    if changed-CHANGED_FUNCTIONS: raise ValueError(f'workload function changed: {changed-CHANGED_FUNCTIONS}')
    return dict(unchanged_function_or_class_count=len(a)-len(changed),
                changed_functions=sorted(changed),
                unchanged_names=sorted(set(a)-changed))

def validate_native(gate: dict, pair: dict):
    if (gate.get('schema_version')!='distillschedule.livewan_two_cycle_timing_gate.v2'
        or gate.get('status')!='PASS' or gate.get('claimable_for_two_cycle_timing') is not True
        or gate.get('cycles')!=2 or gate.get('final_versions')!=dict(generator=2,critic=10,ema=0)
        or gate.get('numerical_review',{}).get('extreme_failure_count')!=0
        or gate.get('numerical_review',{}).get('policy')!='REJECT_EXTREME_ONLY_RECORD_ORDINARY_REVIEW'
        or gate.get('arms',{}).get('pipeline',{}).get('pipeline_variant')!=VARIANT
        or pair.get('status')!='PASS' or pair.get('formal_claimable') is not True
        or pair.get('timing_contract',{}).get('profiled') is not False
        or pair.get('expected_head')!=gate.get('expected_head')):
        raise ValueError('native correctness/timing pair is not the completed C29 source')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native-gate',required=True,type=Path)
    p.add_argument('--native-pair',required=True,type=Path)
    p.add_argument('--expected-head',required=True)
    p.add_argument('--reviewed-diff-sha256',required=True)
    p.add_argument('--output',required=True,type=Path)
    args=p.parse_args()
    gate=json.loads(args.native_gate.read_text());pair=json.loads(args.native_pair.read_text())
    validate_native(gate,pair)
    def git(*a): return subprocess.check_output(['git','-C',str(ROOT),*a])
    if git('rev-parse','HEAD').decode().strip()!=args.expected_head or git('status','--porcelain').strip():
        raise SystemExit('profile source is not the exact clean HEAD')
    parent=gate['expected_head']
    subprocess.run(['git','-C',str(ROOT),'merge-base','--is-ancestor',parent,args.expected_head],check=True)
    paths=set(git('diff','--name-only',parent,args.expected_head).decode().splitlines())
    if paths-ALLOWED: raise SystemExit(f'non-profile change: {sorted(paths-ALLOWED)}')
    diff=git('diff','--binary',parent,args.expected_head)
    diff_sha=hashlib.sha256(diff).hexdigest()
    if diff_sha!=args.reviewed_diff_sha256: raise SystemExit('instrumentation diff not reviewed')
    ast_receipt=verify_worker_ast(git('show',f'{parent}:{WORKER}').decode(),(ROOT/WORKER).read_text())
    for arm in ('baseline','pipeline'):
        data=gate['arms'][arm]
        if sha256(data['trajectory_summary'])!=data['trajectory_summary_sha256']:
            raise SystemExit('trajectory source changed')
        for anchor in data['artifact_binding']['anchors']:
            if sha256(anchor['path'])!=anchor['file_sha256']: raise SystemExit('anchor source changed')
    value=dict(schema_version=SCHEMA,status='PASS_DIAGNOSTIC_ONLY',cycles=2,
        pipeline_variant=VARIANT,profile_authorized=True,formal_timing_authorized=False,
        heads=dict(baseline=parent,pipeline=parent,profile=args.expected_head),
        profile_source=dict(workload_head=parent,profile_head=args.expected_head,instrumentation_only=True,
            diff_paths=sorted(paths),diff_sha256=diff_sha,worker_ast=ast_receipt,
            verification='reviewed_instrumentation_diff_plus_unchanged_math_helpers'),
        native_gate=dict(path=str(args.native_gate),sha256=sha256(args.native_gate)),
        native_pair=dict(path=str(args.native_pair),sha256=sha256(args.native_pair)),
        identity=gate['identity'],seed=gate['seed'],
        claim_boundary='Diagnostic profile only; native C29 50.6243125 s remains unchanged; strict numerics REVIEW.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
    print(json.dumps({'status':value['status'],'head':args.expected_head,'diff_sha256':diff_sha}))

if __name__=='__main__':main()
