"""Opt-in manual-only W6 experiment. NOT an automatic planner candidate."""
from pathlib import Path
import argparse,json
from distillschedule.strategies.h3_ar_assets import H3ARAssets
from distillschedule.strategies.h3_ar_task import H3ARTask,H3ARImplementation
from distillschedule.strategies.h3_ar_worker import run_worker

def main():
    p=argparse.ArgumentParser()
    for name in ('task','assets','capacity-policy','output'):p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--manual-policy',required=True,choices=('off','uncached-rollout-fa3-varlen-v1'))
    p.add_argument('--fa3-kernel-root',type=Path)
    p.add_argument('--execute',action='store_true');a=p.parse_args()
    task=H3ARTask.from_dict(json.loads(a.task.read_text()))
    assets=H3ARAssets.from_dict(json.loads(a.assets.read_text()))
    policy=json.loads(a.capacity_policy.read_text())
    if not a.execute:print('INTENT_ONLY_NO_GPU');return
    run_worker(task,assets,a.output,mode='manual',capacity_policy=policy,
               implementation=H3ARImplementation('zero3-ap'),manual_policy_id=a.manual_policy,
               manual_backend_root=str(a.fa3_kernel_root) if a.fa3_kernel_root else None)
if __name__=='__main__':main()
