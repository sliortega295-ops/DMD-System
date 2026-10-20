"""Execute the real pure-CPU hardware-binding resolver; no training mocks."""
import ast,copy,hashlib,json,os,tempfile,unittest
from pathlib import Path
from typing import Any
SOURCE=Path(os.environ.get("LIVEWAN_TIMING_SUMMARY_UNDER_TEST",
 str(Path(__file__).resolve().parents[1]/"scripts/distillschedule/summarize_livewan_two_cycle_timing_world16.py")))
tree=ast.parse(SOURCE.read_text())
node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="_bound_correctness_hardware")
V=dict(generator=2,critic=10,ema=0)
ns=dict(Any=Any,Path=Path,hashlib=hashlib,json=json,FINAL_VERSIONS=V)
exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(SOURCE),"exec"),ns)
resolve=ns["_bound_correctness_hardware"]
HW=dict(name="h100-8-b16",physical_world_size=8,logical_world_size=16,global_batch=16,
 generator_pool_size=4,score_pool_size=4,pipeline_waves=4,critic_physical_batch_size=2,critic_accumulations_per_commit=2)
HEAD="4d7c589acd01a4a6c90ca141da266b3090df2b68"
IDENTITY=dict(bootstrap_sha256="a"*64,source_receipt_sha256="b"*64,trace_sha256="c"*64)
class HardwareBinding(unittest.TestCase):
 def setup_gate(self,root):
  placement=dict(name="full-resident",teacher_parameter_offload="off",hardware_profile=HW)
  gate=dict(schema_version="distillschedule.livewan_baseline_placement_timing_gate.v1",status="PASS",
   expected_head=HEAD,hardware_profile=HW,baseline_placement=placement,identity=IDENTITY,
   claimable_for_two_cycle_timing=True,cycles=2,final_versions=V,
   numerical_policy="REJECT_EXTREME_ONLY_RECORD_ORDINARY_REVIEW",
   correctness_summary=dict(sha256="d"*64),comparison=dict(sha256="e"*64))
  path=root/"gate.json";path.write_text(json.dumps(gate))
  binding=dict(schema_version=gate["schema_version"],baseline_placement=copy.deepcopy(placement),
   path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
   correctness_summary_sha256="d"*64,trajectory_comparison_sha256="e"*64)
  return binding,gate,path
 def run_resolve(self,binding,root):
  return resolve(binding,expected=HW,expected_head=HEAD,identity=IDENTITY,allowed_root=root)
 def test_serialized_field_still_exact_without_legacy_recovery(self):
  observed,proof=self.run_resolve(dict(hardware_profile=HW),Path("/not-opened"))
  self.assertEqual(observed,HW);self.assertEqual(proof["source"],"SERIALIZED_RECEIPT_FIELD")
 def test_real_hash_bound_legacy_missing_field(self):
  with tempfile.TemporaryDirectory() as d:
   b,g,p=self.setup_gate(Path(d));before=p.read_bytes()
   observed,proof=self.run_resolve(b,Path(d))
   self.assertEqual(observed,HW);self.assertEqual(proof["source"],"HASH_VERIFIED_ORIGINAL_RESIDENT_GATE")
   self.assertNotIn("hardware_profile",b);self.assertEqual(before,p.read_bytes())
 def test_contradictory_serialized_field_is_never_recovered(self):
  with tempfile.TemporaryDirectory() as d:
   b,g,p=self.setup_gate(Path(d));b["hardware_profile"]={"name":"wrong"}
   observed,proof=self.run_resolve(b,Path(d));self.assertNotEqual(observed,HW)
   self.assertEqual(proof["source"],"SERIALIZED_RECEIPT_FIELD")
 def test_original_gate_sha_mismatch(self):
  with tempfile.TemporaryDirectory() as d:
   b,g,p=self.setup_gate(Path(d));p.write_text(p.read_text()+" ")
   self.assertIsNone(self.run_resolve(b,Path(d))[0])
 def test_original_gate_identity_changes_fail_even_with_recomputed_hash(self):
  for field,value in (("hardware_profile",{"name":"a100"}),("expected_head","wrong"),
    ("status","FAIL"),("claimable_for_two_cycle_timing",False),("identity",{}),
    ("cycles",1),("final_versions",{}),("numerical_policy","RELAXED"),
    ("correctness_summary",{"sha256":"different"}),("comparison",{"sha256":"different"})):
   with self.subTest(field=field),tempfile.TemporaryDirectory() as d:
    b,g,p=self.setup_gate(Path(d));g[field]=value;p.write_text(json.dumps(g))
    b["sha256"]=hashlib.sha256(p.read_bytes()).hexdigest()
    self.assertIsNone(self.run_resolve(b,Path(d))[0])
 def test_missing_original_file(self):
  with tempfile.TemporaryDirectory() as d:
   b,g,p=self.setup_gate(Path(d));p.unlink();self.assertIsNone(self.run_resolve(b,Path(d))[0])
 def test_other_schema_cannot_use_legacy_resident_recovery(self):
  with tempfile.TemporaryDirectory() as d:
   b,g,p=self.setup_gate(Path(d));b["schema_version"]="pipeline"
   self.assertIsNone(self.run_resolve(b,Path(d))[0])
 def test_no_gate_escape_from_campaign(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);b,g,p=self.setup_gate(root);allowed=root/"nested";allowed.mkdir()
   self.assertIsNone(self.run_resolve(b,allowed)[0])
if __name__=="__main__":unittest.main()
