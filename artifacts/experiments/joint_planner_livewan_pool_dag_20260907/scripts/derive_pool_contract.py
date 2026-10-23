"""CPU-only derivation beside old native receipts. Does not import Torch."""
import argparse,csv,hashlib,importlib.util,json,subprocess,sys
from collections import Counter
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--module',type=Path,required=True);p.add_argument('--module-sha',required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
assert hashlib.sha256(a.module.read_bytes()).hexdigest()==a.module_sha
spec=importlib.util.spec_from_file_location('distillschedule.search.livewan_pool_contract',a.module)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
assert 'torch' not in sys.modules, 'CPU metadata analysis imported Torch unexpectedly'
from distillschedule.search.native import checked_summary,binding
from distillschedule.search.plan_ir import graph_from_dict
from distillschedule.workloads.best_known.registry import write_new_json
plan=json.loads(a.plan.read_text());out,old_summary=checked_summary(plan)
assert old_summary['status']=='PASS'
raws=[json.loads((out/'operation_calibration'/f'rank_{i:03d}.json').read_text()) for i in range(16)]
c=m.build_pool_contract(raws,expected_head=plan['integration_head'])
gate=m.validate_pool_contract(c,observations=raws)
a.output.mkdir(exist_ok=False)
write_new_json(a.output/'pool-contract.json',c)
g=graph_from_dict(c['graph'])
rows=[dict(id=o.operation_id,pool=o.role,cycle=o.metadata['cycle'],step=o.logical_step,
           family=o.metadata['family'],native_source=o.metadata['native_source_boundary'],
           required_versions=json.dumps(o.required_versions,sort_keys=True),produced_versions=json.dumps(o.produced_versions,sort_keys=True)) for o in g.operations]
with (a.output/'pool-actions.csv').open('x',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
paths=Counter(x['path'] for x in c['messages'])
summary=dict(status='PASS_CPU_SOURCE_GRAPH_REUSING_NATIVE_OBSERVATIONS',gate=gate,
 native_source_head=plan['integration_head'],analysis_module_sha256=a.module_sha,
 native_run_id=plan['run_id'],native_rank_count=16,matched_sample_identities=160,
 node_count=len(g.operations),dependency_count=len(g.dependencies),message_count_per_lane=dict(paths),
 message_bytes_per_lane_by_path={k:sum(x['bytes_per_lane'] for x in c['messages'] if x['path']==k) for k in paths},
 storage_descriptors=len(c['storage']),unmeasured_storage_descriptors=sum(x['observed_bytes'] is None for x in c['storage']),
 serialized_graph='CPU_STRUCTURE_NOT_EXECUTED_NOT_A_MEASURED_PLANNING_PROBLEM',
 native_lowering='NOT_RUN',cost_admission='UNADMITTED',memory_feasibility='UNRESOLVED',new_gpu_runs=0,new_speedup=None,
 sources=[binding(a.plan),binding(out/'bound-summary.json'),*[binding(out/'operation_calibration'/f'rank_{i:03d}.json') for i in range(16)]],
 remaining=c['missing'])
write_new_json(a.output/'summary.json',summary)
write_new_json(a.output/'artifact_sha256.json',{x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in sorted(a.output.iterdir()) if x.is_file()})
print(json.dumps(summary))
