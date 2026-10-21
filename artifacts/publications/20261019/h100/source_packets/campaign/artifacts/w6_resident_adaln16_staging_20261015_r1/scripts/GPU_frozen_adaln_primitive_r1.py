"""One owned H100: actual native full-sized frozen AdaLN vs pinned-CPU staging."""
import contextlib,copy,hashlib,json,pathlib,sys,torch,types
from diffsynth.models.minimax_h3_dit import MiniMaxH3AdalnProj
from runtime.h3_diffsynth_resident_adaln_cpu import FrozenAdaLNCPUStaging
out=pathlib.Path(sys.argv[1]);assert not out.exists()
assert torch.cuda.device_count()==1 and 'H100' in torch.cuda.get_device_name(0)
torch.manual_seed(114)
module=MiniMaxH3AdalnProj(5376,2688,96768,expand_ratio=6,modality_num=3).to(device='cuda',dtype=torch.bfloat16).requires_grad_(False)
reference=copy.deepcopy(module)
role=['student'];policy=FrozenAdaLNCPUStaging([types.SimpleNamespace(adaln_proj=module)],lambda:role[0],count=1)
assert module.linear.weight.device.type=='cpu' and module.linear.weight.is_pinned()
assert torch.equal(reference.linear.weight.detach().cpu(),module.linear.weight)
assert torch.equal(reference.linear.bias.detach().cpu(),module.linear.bias)
torch.cuda.reset_peak_memory_stats();checks=[];shape_cases=set();grad_cases=set()
for name,count in (('student',20),('fake_score',6),('teacher',2)):
 role[0]=name
 for i in range(count):
  m=1+i%2;x=torch.randn(m,2688,device='cuda',dtype=torch.bfloat16);shape_cases.add(m)
  enabled=name!='teacher' and i%2==0;grad_cases.add(enabled)
  with torch.set_grad_enabled(enabled),torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
   expected=reference(x);actual=module(x)
  checks.append(all(torch.equal(a,b) and not b.requires_grad for a,b in zip(expected,actual)))
receipt=policy.finish_cycle();assert all(checks) and len(checks)==28
assert receipt['canonical_CPU_bytes']==520418304 and receipt['counters']['H2D_bytes']==28*520418304
assert shape_cases=={1,2} and grad_cases=={False,True}
torch.cuda.synchronize()
r=dict(status='PASS_ACTUAL_FULL_SHAPE_FROZEN_ADALN_CUDA_STAGED_BITWISE',scope='one520MB nativeAdaLN projection, not33Bmodelcapacity orLoRAgradientproof',output_checks=len(checks),all_bitwise=True,shapes=sorted(shape_cases),grad_mode_cases=sorted(grad_cases),dtype='BF16',exact_CPU_master_values_equal=True,pinned_CPU=True,source_module_sha256=hashlib.sha256(pathlib.Path(sys.modules[MiniMaxH3AdalnProj.__module__].__file__).read_bytes()).hexdigest(),staging=receipt,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),driver_free_after=torch.cuda.mem_get_info()[0])
policy.close()
with out.open('x') as f:json.dump(r,f,indent=2)
print(json.dumps(r))
