"""Execute the exact pre-import cache/failure helpers without importing Torch."""
import ast,os,pathlib,tempfile,types,unittest
ROOT=pathlib.Path(__file__).resolve().parents[1]
SOURCE=ROOT/"scripts/distillschedule/probe_wan13b_two_cycle_timing_world16.py"
def native_norm_identity_fixture(value):
    return value
def helpers():
    tree=ast.parse(SOURCE.read_text())
    names={"_configure_rank_private_triton_cache","_write_failure_receipt_and_maybe_exit"}
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    ns=dict(os=os,Path=pathlib.Path,Any=object)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(SOURCE),"exec"),ns)
    return ns
class TestCache(unittest.TestCase):
    def test_default_off_preserves_existing_cache(self):
        env=dict(TRITON_CACHE_DIR="/existing")
        x=helpers()["_configure_rank_private_triton_cache"](env)
        self.assertEqual(x,dict(enabled=False,fail_fast_on_error=False));self.assertEqual(env["TRITON_CACHE_DIR"],"/existing")
    def test_real_rank_paths_are_unique_and_writable(self):
        base=pathlib.Path("/tmp/lyy-experiments/DMD-System/cache_CPU_gate");base.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as root:
            paths=[]
            for rank in range(8):
                env=dict(DISTILLSCHEDULE_WAN13B_RANK_CACHE_ROOT=root,WORLD_SIZE="8",LOCAL_RANK=str(rank),DISTILLSCHEDULE_WORKER_FAILFAST_ON_ERROR="1")
                x=helpers()["_configure_rank_private_triton_cache"](env);p=pathlib.Path(x["path"]);(p/"metadata.json").write_text(str(rank));paths.append(p)
                self.assertTrue(x["fail_fast_on_error"]);self.assertEqual(env["TRITON_CACHE_DIR"],str(p))
            self.assertEqual(len(set(paths)),8)
            self.assertEqual([(p/"metadata.json").read_text() for p in paths],list(map(str,range(8))))
    def test_unsafe_or_unqualified_paths_fail(self):
        fn=helpers()["_configure_rank_private_triton_cache"]
        for root,world,rank in (("/tmp/other","8","0"),("/tmp/lyy-experiments/DMD-System/x","16","0"),("/tmp/lyy-experiments/DMD-System/x","8","8")):
            with self.assertRaises(ValueError):fn(dict(DISTILLSCHEDULE_WAN13B_RANK_CACHE_ROOT=root,WORLD_SIZE=world,LOCAL_RANK=rank))
    def test_invalid_error_flag_fails_before_any_import(self):
        with self.assertRaises(ValueError):helpers()["_configure_rank_private_triton_cache"](dict(DISTILLSCHEDULE_WORKER_FAILFAST_ON_ERROR="yes"))
    def test_failure_is_written_before_noncollective_exit(self):
        ns=helpers();events=[];ns["_atomic_json"]=lambda p,r:events.append("written")
        def exit(code):events.append(("exit",code));raise SystemExit(code)
        ns["os"]=types.SimpleNamespace(_exit=exit)
        with self.assertRaises(SystemExit):ns["_write_failure_receipt_and_maybe_exit"]("out",{},fail_fast=True)
        self.assertEqual(events,["written",("exit",1)])
    def test_default_failure_behavior_unchanged(self):
        ns=helpers();events=[];ns["_atomic_json"]=lambda p,r:events.append("written")
        ns["os"]=types.SimpleNamespace(_exit=lambda c:self.fail("unrequested exit"))
        ns["_write_failure_receipt_and_maybe_exit"]("out",{},fail_fast=False);self.assertEqual(events,["written"])
    def test_application_cycle_literal_unchanged(self):
        import hashlib
        text=SOURCE.read_text();tree=ast.parse(text);node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=="_run_workload_cycle")
        source="\n".join(text.splitlines()[node.lineno-1:node.end_lineno])+"\n"
        # Bound against actual original function, not a manually invented hash.
        import subprocess
        old=subprocess.check_output(["git","show","d07876e6f95848cec8cf6bc85848fe34003763d8:"+str(SOURCE.relative_to(ROOT))],cwd=ROOT,text=True)
        original=next(x for x in ast.parse(old).body if isinstance(x,ast.FunctionDef) and x.name=="_run_workload_cycle")
        expected="\n".join(old.splitlines()[original.lineno-1:original.end_lineno])+"\n"
        self.assertEqual(hashlib.sha256(source.encode()).hexdigest(),hashlib.sha256(expected.encode()).hexdigest())
    def test_cache_config_precedes_torch_import(self):
        text=SOURCE.read_text();self.assertLess(text.index("_RANK_PRIVATE_CACHE ="),text.index("import torch\n"))
    def test_canary_uses_production_offline_import_before_native_norm(self):
        source=ROOT/"scripts/distillschedule/canary_wan13b_rank_private_cache.py"
        tree=ast.parse(source.read_text())
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="load_native_rmsnorm")
        statements=node.body[1:]
        self.assertIsInstance(statements[0],ast.ImportFrom)
        self.assertEqual(statements[0].module,"runtime.critic_pool_helios")
        self.assertEqual(statements[0].names[0].name,"configure_offline_helios")
        self.assertEqual(ast.unparse(statements[1]),"configure_offline_helios()")
        self.assertIsInstance(statements[2],ast.ImportFrom)
        self.assertEqual(statements[2].module,"helios.modules.helios_kernels.triton_norm")
        self.assertEqual(ast.unparse(statements[3]),"return flash_rms_layernorm")
        self.assertIn("flash_rms_layernorm = load_native_rmsnorm()",source.read_text())
    def test_canary_identity_unwraps_decorator_and_rejects_other_source(self):
        import functools,hashlib,inspect
        source=ROOT/"scripts/distillschedule/canary_wan13b_rank_private_cache.py"
        tree=ast.parse(source.read_text())
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="native_source_receipt")
        expected=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()
        ns=dict(pathlib=pathlib,hashlib=hashlib,NATIVE_NORM_SHA256=expected)
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),"exec"),ns)
        @functools.wraps(native_norm_identity_fixture)
        def wrapper(value):
            return native_norm_identity_fixture(value)
        receipt=ns["native_source_receipt"](wrapper)
        self.assertEqual(receipt["file_sha256"],expected)
        self.assertEqual(receipt["callable_source_sha256"],hashlib.sha256(inspect.getsource(native_norm_identity_fixture).encode()).hexdigest())
        ns["NATIVE_NORM_SHA256"]="0"*64
        with self.assertRaisesRegex(ValueError,"NATIVE_RMSNORM_SOURCE_IDENTITY_CHANGED"):
            ns["native_source_receipt"](wrapper)
if __name__=="__main__":unittest.main()
