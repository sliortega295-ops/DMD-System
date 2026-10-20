from pathlib import Path
import json,hashlib,sys,subprocess,ast,shutil
R=Path('/home/jiangzhikun/yongyan_liu/Helios-H100');C=R/'research_artifacts/five_workload_h100_20260930';O=C/'h100_resident_baselines_20261014_r1';old=O/'artifacts/h3_three_arm_prepare_r4';new=O/'artifacts/h3_named_boundary_prepare_r3'
receipt=json.load(open(O/'manifests/h3_named_boundary_contract_CPU_r3.json'));assert receipt['status']=='PASS_ACTUAL_PRIVATE_CPU_TESTS_AND_UNCHANGED_MATH_SOURCE'
repo=R/'repos/H3-DMD-for-A100-h100-resident-boundary-20261014-r3'
config='configs/minimax_h3_grid1000_h100_resident_boundary_cpu_b16.yaml'
new.mkdir(exist_ok=False);(new/'scripts').mkdir()
for name in ('authorization.json','real_cache_CPU_receipt.json'):
 assert (old/name).exists();shutil.copyfile(old/name,new/name)
policy=json.load(open(old/'policy_freeze.json'));policy.update(h3_head='d81ebb2014af9ea3a84011e3be198818e63cb1e7',h100_config_sha256=hashlib.sha256((repo/config).read_bytes()).hexdigest(),hardware_matrix=['resident-boundary-cpu'],control_label='NAMED_RESIDENT_BOUNDARY_CPU_CAPACITY_REFERENCE_NOT_PURE_PLACEMENT',new_optimization='NONE_EXISTING_BOUNDARY_POLICY_AND_NEW_ADMISSION_PROFILE')
for key in ('control_environment','candidate_environment'):
 policy[key]=dict(policy['control_environment'],H3_ACTIVATION_POLICY='checkpoint_boundary_cpu',H3_HARDWARE_PROFILE='h100-8-b16-resident-boundary-cpu')
with (new/'policy_freeze.json').open('x') as f:json.dump(policy,f,indent=2)
with (new/'source_policy_binding.json').open('x') as f:json.dump({'policy_sha256':hashlib.sha256((new/'policy_freeze.json').read_bytes()).hexdigest(),'runtime_contract_source_head':policy['h3_head'],'origin_bootstrap_source':'a4c289757d0ecc8c32eb6ea27021f22cd04d94fd','origin_bootstrap_config_sha256':'b001f2f517b854c34ca68cccce23422499d95b7a3ad27072b895a502498ca6be','unchanged_math_proof':str(O/'manifests/h3_named_boundary_contract_CPU_r3.json')},f,indent=2)
sys.path.insert(0,str(O/'scripts'))
import run_h3_named_resident_boundary_r3 as runner
g=runner.load_guard();g.verify_clean_sources();g.validate_policy(policy)
env=g.make_environment(policy['control_environment'],new)
env['PYTHONPATH']=str(O/'scripts')+':'+env['PYTHONPATH']
# Resolved structured config and actual profile parser gate, no model/cuda.
code="from lightx2v_train.runtime import load_config;from h3_a100.hardware_profile import resolve_hardware_contract;import torch,sys;c=load_config(sys.argv[1]);h=resolve_hardware_contract(c,actual_world_size=8);assert h.enabled and h.logical_ranks(7)==(7,15);assert c['training']['parameter_placement']=='resident';assert not torch.cuda.is_initialized();print('PASS_ACTUAL_NEW_NAMED_CONFIG_IMPORT_AND_MAPPING_NO_MODEL')"
env['H3_LOGICAL_RANK_REPLAY_MANIFEST']=str(C/'h3_three_arm_common_bootstrap_build_20261014_r2/bootstrap/manifest.json')
with (O/'logs/h3_named_boundary_resolved_config_CPU_r3.log').open('x') as f:subprocess.run([str(g.PYTHON),'-c',code,str(repo/config)],env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=180)
with (O/'logs/h3_named_boundary_runner_CPU_r3.log').open('x') as f:subprocess.run([str(g.PYTHON),str(O/'scripts/run_h3_named_resident_boundary_r3.py'),'--run-id','h3_named_boundary_registered_runner_CPU_prepare_20261014_r3'],stdout=f,stderr=subprocess.STDOUT,check=True,timeout=180)
print(json.dumps({'status':'PASS_CPU_NAMED_BOUNDARY_RUNNER','source':g.HEAD,'runtime_config_sha':g.CONFIG_SHA,'GPU':'NOT_RUN'}))
