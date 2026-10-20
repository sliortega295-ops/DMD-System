import ast,pathlib,subprocess,json,hashlib,datetime,collections
R=pathlib.Path("/home/jiangzhikun/yongyan_liu/Helios-H100/repos/DMD-System-h100-w6-lora-unit-inplace-20261015-r1");P=pathlib.Path("/home/jiangzhikun/yongyan_liu/Helios-H100/research_artifacts/five_workload_h100_20260930/h100_resident_baselines_20261014_r1/artifacts/w6_resident_adaln16_staging_20261015_r1")
old='1abd94656420880f71b0f8adfdcc2974952c33c2';new='0419b96b9ed9a8dd76737042e6c5020c80f6673a';name='scripts/distillschedule/probe_h3_diffsynth_h100_world8.py'
assert subprocess.check_output(['git','-C',str(R),'rev-parse','HEAD'],text=True).strip()==new
a=subprocess.check_output(['git','-C',str(R),'show',old+':'+name],text=True);b=(R/name).read_text()
counter=collections.Counter()
def attr(n):return any(isinstance(x,ast.Attribute) and x.attr=='resident_adaln_cpu_blocks' and isinstance(x.value,ast.Name) and x.value.id=='args' for x in ast.walk(n))
def staging(n):return any(isinstance(x,ast.Name) and x.id=='adaln_staging' for x in ast.walk(n))
class DefaultOff(ast.NodeTransformer):
 def visit_Expr(self,node):
  v=node.value
  if isinstance(v,ast.Call) and any(isinstance(x,ast.Constant) and x.value=='--resident-adaln-cpu-blocks' for x in v.args):
   kw={k.arg:k.value for k in v.keywords};assert ast.literal_eval(kw['default'])==0 and ast.literal_eval(kw['choices'])==(0,16);counter['parser_default0']+=1;return None
  if isinstance(v,ast.Call) and isinstance(v.func,ast.Name) and v.func.id=='require' and attr(v.args[0]):
   assert len(v.args)==2 and ast.literal_eval(v.args[1])=='ADALN_CPU_IS_NAMED_RESIDENT_CAPACITY_FALLBACK';counter['default_guard']+=1;return None
  return self.generic_visit(node)
 def visit_Assign(self,node):
  if len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='adaln_staging':
   assert isinstance(node.value,ast.Constant) and node.value.value is None;counter['default_none_assignment']+=1;return None
  return self.generic_visit(node)
 def visit_If(self,node):
  if attr(node.test):
   assert not node.orelse;counter['never_entered_default0_branch']+=1;return None
  if staging(node.test):
   assert ast.unparse(node.test)=='adaln_staging is not None' and not node.orelse;counter['never_entered_default_none_branch']+=1;return None
  return self.generic_visit(node)
 def visit_Dict(self,node):
  for i,key in enumerate(node.keys):
   if isinstance(key,ast.Constant) and key.value=='parameter_offload' and attr(node.values[i]):
    v=node.values[i];assert isinstance(v,ast.Call) and isinstance(v.func,ast.Name) and v.func.id=='bool'
    node.values[i]=ast.Constant(False);counter['metadata_bool0_to_false']+=1
  return self.generic_visit(node)
norm=DefaultOff().visit(ast.parse(b))
expected=dict(parser_default0=1,default_guard=1,default_none_assignment=1,never_entered_default0_branch=2,never_entered_default_none_branch=3,metadata_bool0_to_false=1)
assert dict(counter)==expected,dict(counter)
assert ast.dump(norm,include_attributes=False)==ast.dump(ast.parse(a),include_attributes=False),'DEFAULT_OFF_WORKER_AST_NOT_IDENTICAL'
proof=dict(status='PASS_DEFAULT_OFF_WHOLE_WORKER_AST_EQUIVALENCE_CPU_ONLY',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),old=old,new=new,source=name,old_sha256=hashlib.sha256(a.encode()).hexdigest(),new_sha256=hashlib.sha256(b.encode()).hexdigest(),constant_branches=expected,scope='Only absent --resident-adaln-cpu-blocks/default0 and defaultNone bookkeeping normalized. Complete worker AST matchesold; no loss/forward/backward/RNG/checkpoint/order transformation.',new_full_model_strict_parity='NOT_RUN',prior_exact_C1_training_head=old,GPU='NOT_RUN',new_model_or_checkpoint_values=False)
with (P/'manifests/default_off_nonresident_source_equivalence_r1.json').open('x') as f:json.dump(proof,f,indent=2)
print(json.dumps(proof))
