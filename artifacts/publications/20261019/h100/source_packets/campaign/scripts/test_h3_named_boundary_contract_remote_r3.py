from pathlib import Path
import sys,subprocess,json,hashlib
R=Path('/home/jiangzhikun/yongyan_liu/Helios-H100');O=R/'research_artifacts/five_workload_h100_20260930/h100_resident_baselines_20261014_r1'
sys.path.insert(0,str(O/'scripts'));import run_h3_h100_three_arm_r4 as base
g=base.load_guard();policy=g.read_hashed(g.PREPARED/'policy_freeze.json',g.POLICY_SHA)
N=R/'repos/H3-DMD-for-A100-h100-resident-boundary-20261014-r3'
assert subprocess.check_output(['git','-C',str(N),'rev-parse','HEAD'],text=True).strip()=='d81ebb2014af9ea3a84011e3be198818e63cb1e7'
assert not subprocess.check_output(['git','-C',str(N),'status','--porcelain'],text=True).strip()
env=g.make_environment(policy['control_environment'],O/'artifacts/h3_three_arm_prepare_r4')
env['PYTHONPATH']=str(N)+':'+env['PYTHONPATH'];env['CUDA_VISIBLE_DEVICES']=''
with (O/'logs/h3_named_boundary_contract_CPU_r3.log').open('x') as f:
 subprocess.run([str(g.PYTHON),'-m','pytest','-q',str(N/'tests/test_h100_resident_boundary_profile.py'),str(N/'tests/test_h100_hardware_profile.py'),str(N/'tests/test_checkpoint_boundary_offload.py')],env=env,cwd=N,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=180)
# New compatibility profile is the only core source modification. Initializer,
# sample/RNG restore bodies and original model/optimizer/training math unchanged.
files=('h3_a100/model.py','h3_a100/trainer.py','h3_a100/trainer_loop.py','h3_a100/grid_adaln.py','h3_a100/grid_entrypoint.py','h3_a100/fresh_h100_bootstrap.py','h3_a100/checkpoint_boundary_offload.py')
rows=[]
for file in files:
 old=g.REPO/file;new=N/file
 assert old.read_bytes()==new.read_bytes(),file
 rows.append({'path':file,'sha256':hashlib.sha256(new.read_bytes()).hexdigest(),'status':'UNCHANGED_BYTE_FOR_BYTE'})
x={'status':'PASS_ACTUAL_PRIVATE_CPU_TESTS_AND_UNCHANGED_MATH_SOURCE','source_head':'d81ebb2014af9ea3a84011e3be198818e63cb1e7','old_source':'9bfd3eda959bd0c8415fe71ba64d3ad5f7e93e71','math_files':rows,'GPU':'NOT_RUN','old_none_profile':'still rejects boundaryoverride','new_profile':'explicit resident+boundaryCPU,segment1,SP1,8physical16logicalGAS2'}
with (O/'manifests/h3_named_boundary_contract_CPU_r3.json').open('x') as f:json.dump(x,f,indent=2)
print(json.dumps(x))
