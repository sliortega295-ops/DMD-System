"""Actual private AdaLN CPU expression tests; not CUDA capacity proof."""
import ast,copy,unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from runtime.h3_diffsynth_resident_adaln_cpu import FrozenAdaLNCPUStaging
from diffsynth.models.minimax_h3_dit import MiniMaxH3AdalnProj

class CPUFixtureOnly(FrozenAdaLNCPUStaging):
    # No production bypass flag: this subclass exists ONLY in the test file.
    def _copy_master(self, parameter):
        return parameter.detach().cpu().clone()
    def _check_input(self, value):
        if value.device.type != "cpu" or value.requires_grad:
            raise RuntimeError("frozen timestep input")

def projections():
    torch.manual_seed(7)
    return [SimpleNamespace(adaln_proj=MiniMaxH3AdalnProj(8,4,144,expand_ratio=6,modality_num=3)
            .to(torch.bfloat16).requires_grad_(False)) for _ in range(2)]

def drive(policy, blocks):
    for role, number in (("student",20),("fake_score",6),("teacher",2)):
        policy.current_role=role
        for _ in range(number):
            for block in blocks:
                block.adaln_proj(torch.ones(2,4,dtype=torch.bfloat16))

class FrozenAdaLNTests(unittest.TestCase):
    def make(self):
        blocks=projections()
        p=CPUFixtureOnly(blocks,lambda:p.current_role,count=2)
        p.current_role="student"
        return blocks,p

    def test_disabled_does_not_inspect_or_change_modules(self):
        value=object();p=FrozenAdaLNCPUStaging(value,None,count=0)
        self.assertFalse(p.enabled);self.assertEqual(p.finish_cycle()["counters"],{});p.close()

    def test_real_private_expression_bitwise_roles_and_hooks(self):
        blocks=projections();reference=copy.deepcopy(blocks);hooks=[]
        originals=[]
        for b in blocks:
            m=b.adaln_proj.linear
            m.register_forward_pre_hook(lambda *a:hooks.append("pre"))
            m.register_forward_hook(lambda *a:hooks.append("post"))
            originals.append(m)
        p=CPUFixtureOnly(blocks,lambda:p.current_role,count=2)
        for role,number in (("student",20),("fake_score",6),("teacher",2)):
            p.current_role=role
            for _ in range(number):
                x=torch.randn(2,4,dtype=torch.bfloat16)
                with torch.autocast("cpu",dtype=torch.bfloat16):
                    for a,b in zip(reference,blocks):
                        expected=a.adaln_proj(x);actual=b.adaln_proj(x)
                        self.assertTrue(all(torch.equal(u,v) for u,v in zip(expected,actual)))
                        self.assertTrue(all(not u.requires_grad for u in actual))
        self.assertEqual([b.adaln_proj.linear for b in blocks],originals)
        self.assertEqual(len(hooks),112)
        r=p.finish_cycle();self.assertEqual(r["counters"]["linear_calls"],56)
        self.assertEqual(r["counters"]["H2D_count"],112)
        self.assertEqual(r["calls_by_role"],dict(student=40,fake_score=12,teacher=4))
        self.assertFalse(r["parameter_values_or_dtype_changed"])

    def test_production_refuses_CPU_instead_of_mocking_GPU(self):
        blocks=projections()
        with self.assertRaisesRegex(ValueError,"already loaded CUDA"):
            FrozenAdaLNCPUStaging(blocks,lambda:"student",count=1)

    def test_trainable_weight_is_not_offloaded(self):
        blocks=projections();weight=blocks[0].adaln_proj.linear.weight
        weight.requires_grad_(True)
        with self.assertRaisesRegex(ValueError,"only frozen"):
            CPUFixtureOnly(blocks,lambda:"student",count=2)
        self.assertIs(blocks[0].adaln_proj.linear.weight,weight)

    def test_unknown_source_is_rejected_before_mutation(self):
        wrong=[SimpleNamespace(adaln_proj=torch.nn.Linear(4,8))]
        with self.assertRaisesRegex(ValueError,"source identity"):
            CPUFixtureOnly(wrong,lambda:"student",count=1)

    def test_no_differentiable_timestep_path_may_stage(self):
        blocks,p=self.make()
        with self.assertRaisesRegex(RuntimeError,"frozen timestep"):
            blocks[0].adaln_proj(torch.ones(2,4,dtype=torch.bfloat16,requires_grad=True))

    def test_immutable_cpu_parameter_version(self):
        blocks,p=self.make();drive(p,blocks)
        with torch.no_grad():blocks[0].adaln_proj.linear.weight.add_(1)
        with self.assertRaisesRegex(RuntimeError,"mutated"):p.finish_cycle()

    def test_close_does_not_rematerialize_and_forbids_reuse(self):
        blocks,p=self.make();p.close()
        with self.assertRaisesRegex(RuntimeError,"after teardown"):
            blocks[0].adaln_proj(torch.ones(2,4,dtype=torch.bfloat16))

class Admission(unittest.TestCase):
    def guard(self,**kw):
        defaults=dict(resident_adaln_cpu_blocks=16,placement="resident",checkpoint_boundary_cpu=True,
                      attention_backend="native",lora_input_cast_elision=True,lora_unit_inplace=True)
        defaults.update(kw)
        worker=Path(__file__).resolve().parents[1]/"scripts/distillschedule/probe_h3_diffsynth_h100_world8.py"
        tree=ast.parse(worker.read_text())
        main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="main")
        nodes=[n for n in main.body if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and
               any(isinstance(a,ast.Constant) and a.value=="ADALN_CPU_IS_NAMED_RESIDENT_CAPACITY_FALLBACK" for a in n.value.args)]
        self.assertEqual(len(nodes),1)
        def require(x,message):
            if not x:raise ValueError(message)
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(worker),"exec"),
             dict(args=SimpleNamespace(**defaults),require=require))
    def test_only_explicit_native_resident_fallback(self):
        self.guard()
        self.guard(resident_adaln_cpu_blocks=0,placement="fsdp")
        for extra in (dict(placement="fsdp"),dict(placement="candidate"),dict(checkpoint_boundary_cpu=False),
                      dict(attention_backend="fa3"),dict(lora_input_cast_elision=False),dict(lora_unit_inplace=False)):
            with self.subTest(extra=extra),self.assertRaises(ValueError):self.guard(**extra)

if __name__=="__main__":unittest.main()
