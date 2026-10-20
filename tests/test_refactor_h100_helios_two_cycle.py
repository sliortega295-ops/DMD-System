"""CPU protocol fixtures, not GPU proof; actual 8fc logical helpers run by AST.

The three native helper bodies and MicrobatchCommitEngine are unmodified source.
Only inner model computation, tensors, CUDA and asset payloads are test doubles.
"""
import __future__
import ast
import copy
from contextlib import nullcontext
import importlib.util
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from distillschedule.runtime import h100_helios_two_cycle as c2
from distillschedule.search.hardware_helios import _native_control, ARMS


def source_functions(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec",
                 flags=__future__.annotations.compiler_flag), namespace)


class ModeTests(unittest.TestCase):
    def test_default_off_and_unknown_modes_fail_closed(self):
        self.assertEqual(c2.requested_mode({}), "off")
        for mode in ("c2", "auto", "resume", "on"):
            with self.assertRaises(ValueError):
                c2.requested_mode({c2.MODE_ENV: mode})


@unittest.skipUnless(os.environ.get("DMD_HELIOS_H100_NATIVE_ROOT"), "explicit native source required")
class NativeProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(os.environ["DMD_HELIOS_H100_NATIVE_ROOT"]).resolve()
        cls.inventory = _native_control(cls.root, "inventory", {"arms": list(ARMS)})
        path = cls.root / "src/distillschedule/runtime/hardware_profile.py"
        spec = importlib.util.spec_from_file_location("native_h100_profile_cpu_fixture", path)
        cls.hardware_module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cls.hardware_module
        spec.loader.exec_module(cls.hardware_module)
        cls.audit_functions = {"_require": c2._require, "digest": c2._digest, "re": re}
        source_functions(cls.root / c2.AUDITOR,
            ("_operations", "entry_identity", "_check_rollout", "_audit_persistence"), cls.audit_functions)

    def context(self, *, arm=ARMS[2], rank=0, mode="timing"):
        frozen = copy.deepcopy(self.inventory[arm]["frozen"])
        state = SimpleNamespace(events=[], backward=[], constructs=[], pending=set(), outputs=[])
        controls = {}
        for slot in range(16):
            trace = {"trace_hash": c2._digest(["CPU_FIXTURE_ONLY", slot]), "operations": [], "tensors": {}}
            for role, steps in (("generator", (10,)), ("critic", c2.STEPS)):
                for step in steps:
                    prefix = f"step/{step}/{role}"
                    names = ["continuous_history", "history_short", "history_mid", "history_long", "indices_hidden_states",
                        "indices_history_short", "indices_history_mid", "indices_history_long", "entry_noise", "rollout_noise",
                        "block_noise/1", "block_noise/2"] + (["dmd_timestep", "dmd_noise"] if role == "generator" else ["critic_timestep", "critic_noise"])
                    for key in ["sample/0/text_embedding", "sample/0/attention_mask"] + [prefix + "/" + n for n in names]:
                        trace["tensors"][key] = {"sha256": c2._digest([slot, key])}
                    stages = [{"stage": s, "call_count": n, "denoising_exit": n - 1,
                               "latent_shape": [1, 16, 9, 12 * 2**s, 20 * 2**s]} for s, n in enumerate((1, 2, 1))]
                    trace["operations"].append({"key": prefix + "/rollout", "control": {"pyramid_stage_call_sequence": stages, "call_count": 4}})
            controls[slot] = trace
        auditor = SimpleNamespace(**{name: self.audit_functions[name] for name in
            ("entry_identity", "_check_rollout", "_audit_persistence")},
            FLASH_SHA="c2acd487c5d1d77b7f6721eabc0fee4f8c5c8f9e682aadf43bfa99b394eecba7",
            freeze_policy=lambda root, variant, arm: frozen,
            trace_controls=lambda path, variant, sha: controls)

        class Engine:
            def __init__(self, role):
                state.constructs.append(role)
                self.role, self.global_steps, self.module = role, c2.INITIAL[role], object()
                self.optimizer = SimpleNamespace(step=lambda: state.events.append(role + ".optimizer"))
                self.lr_scheduler = SimpleNamespace(step=lambda: state.events.append(role + ".scheduler"))
            def backward(self, loss):
                state.backward.append((self.role, loss))
            def set_gradient_accumulation_boundary(self, *, is_boundary):
                self.boundary = is_boundary
            def step(self):
                self.optimizer.step(); self.lr_scheduler.step(); self.global_steps += 1

        class EMA:
            def __init__(self):
                state.constructs.append("ema"); self.optimization_step = 10; self.last_transfer_bytes = 0
            def step(self):
                self.optimization_step += 1; state.events.append("ema"); return .9
            def checksum(self):
                return {"fixture_scope": "CPU_ONLY", "optimization_step": self.optimization_step}

        def rollout(slot, step, role):
            value = {"call_count": 4, "logical_call_count": 4, "stage_call_count": {0: 1, 1: 2, 2: 1},
                "logical_stage_call_count": {0: 1, 1: 2, 2: 1}, "dummy_call_count": 0, "calls": [],
                "output_dtype": "torch.bfloat16", "output_shape": [1, 16, 9, 48, 80],
                "output_requires_grad": role == "generator", "synchronize_after_rollout": not frozen["policy"]["elide_explicit_rollout_sync"]}
            selected = []
            stages = self.audit_functions["_operations"](controls[slot])[f"step/{step}/{role}/rollout"]["control"]["pyramid_stage_call_sequence"]
            for stage in stages:
                for denoise in range(stage["call_count"]):
                    grad = role == "generator" and denoise == stage["denoising_exit"]
                    index = len(value["calls"])
                    value["calls"].append({"call_index": index, "stage": stage["stage"], "denoise_call_in_stage": denoise,
                        "hidden_states_shape": stage["latent_shape"], "status": "success", "grad_enabled": grad, "output_requires_grad": grad})
                    if grad: selected.append(index)
            identity = auditor.entry_identity(controls[slot], variant="no-gan", slot=slot, step=step, role=role)
            if role == "generator":
                value.update(source_rank=slot, operation_trace_hash=controls[slot]["trace_hash"], trace_identity_sha256=identity,
                    selected_call_indices=selected, empty_cache_before_rollout=not frozen["policy"]["elide_explicit_rollout_sync"])
            else:
                value.update(batch_size=1, logical_steps=[step], member_trace_identities=[identity], trace_key=f"static-fsdp2-p0/batched/{step}")
            return value

        def generator(*, rank, device, generator_engine, critic_engine, trace, negative, collect_diagnostics,
                      diagnostics, policy, score_pack_parity_check, role_swap, expect_optimizer_commit):
            state.events.append(("g", rank, generator_engine.global_steps))
            generator_engine.backward(4); generator_engine.step()
            return {"logical_step": 10, "source_rank": rank, "physical_generator_step_after": generator_engine.global_steps,
                "optimizer_committed": expect_optimizer_commit, "logical_score_requests": 3, "physical_score_forwards": 3,
                "real_score_pair_packed": False, "rollout": rollout(rank, 10, "generator")}

        def precompute(*, rank, steps, device, generator_engine, trace, policy, collect_diagnostics, diagnostics, rollout_parity_check):
            state.events.append(("rollout", rank, generator_engine.global_steps))
            outputs = {step: object() for step in steps}
            state.outputs.append(outputs)
            for step in steps: state.pending.add((rank, step))
            persistence = {"enabled": policy.generator_nograd_persistent_unshard, "rollout_batch_count": 5,
                "scope": "GENERATOR_NOGRAD_ROLLOUTS_ONLY", "expected_fsdp_module_count": 41, "fsdp_module_count": 41,
                "parameter_group_count": 41, "set_no_reshard_count": 41, "restore_reshard_policy_count": 41,
                "explicit_reshard_count": 41, "fully_unsharded_before_exit_count": 41, "unsharded_after_exit_count": 0,
                "restored": True, "flush_allocator_cache": True, "pre_empty_cache_count": 1, "post_empty_cache_count": 1,
                "rollout_memory": [{"fixture_scope": "CPU_ONLY"} for _ in steps]}
            return outputs, [rollout(rank, step, "critic") for step in steps], persistence

        def critic(*, rank, step, x0, device, critic_engine, trace, collect_diagnostics, diagnostics):
            state.pending.remove((rank, step)); state.events.append(("c", rank, step, critic_engine.global_steps))
            critic_engine.backward(6); critic_engine.step()
            return {"logical_step": step, "source_rank": rank, "physical_critic_step_after": critic_engine.global_steps,
                    "score_forwards": 1, "generator_rollout_precomputed": True}

        namespace = {"MicrobatchCommitEngine": self.hardware_module.MicrobatchCommitEngine, "nullcontext": nullcontext,
            "_run_nogan_generator_update_p0": generator, "_precompute_critic_rollouts": precompute,
            "_run_nogan_critic_precomputed": critic}
        source_functions(self.root / c2.WORKER, [*c2.HELPERS, "_aggregate_logical_records"], namespace)
        p0 = SimpleNamespace(__file__=str(self.root / c2.WORKER), **{name: namespace[name] for name in c2.HELPERS},
            hardware_profile=self.hardware_module.hardware_profile,
            torch=SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda device: state.events.append("sync"))),
            dist=SimpleNamespace(barrier=lambda: state.events.append("barrier")))
        args = SimpleNamespace(variant="no-gan", parameter_placement="fsdp2", ema_device="cuda", profile_mode=False,
            profile_diagnostics=False, operation_source=False, critic_only_step=None, timing_mode=mode == "timing",
            trace_bundle=Path("/CPU_FIXTURE_ONLY/trace"), **{k: frozen["policy"][k] for k in c2.ARG_POLICY_FIELDS})
        trace = SimpleNamespace(bundle_hash=frozen["trace_hash"], logical_steps=c2.STEPS,
            receipt=lambda: {"critic_entries_still_cached": len(state.pending), "fixture_scope": "CPU_ONLY"})
        result = {"rank": rank, "world_size": 8, "contract": frozen["contract"], "p0_policy": frozen["policy"],
            "evidence_scope": "CPU_PROTOCOL_FIXTURE_ONLY", "identities": {"trace_bundle_hash": frozen["trace_hash"]},
            "attention_backend": {"status": "PASS", "requested": "local-fa2", "runtime": "local-fa2", "extension_sha256": auditor.FLASH_SHA},
            "hardware_profile": {"name": "h100-8-b16", "physical_world_size": 8, "logical_global_batch": 16,
                "logical_b1_microbatches_per_rank": 2, "logical_slots": [rank, rank + 8], "loss_scale_per_logical_b1": .5,
                "rank_mapping": "slot = physical_rank + microbatch * physical_world_size"}}
        kwargs = dict(p0=p0, args=args, policy=SimpleNamespace(**frozen["policy"]), result=result, rank=rank, device="CPU_FIXTURE_ONLY",
            logical=SimpleNamespace(manifest_hash=frozen["logical_hash"], component=lambda role: {"optimizer_step": c2.INITIAL[role]}),
            trace=trace, negative=object(), cfg=None, generator_engine=Engine("generator"), critic_engine=Engine("critic"), ema=EMA(),
            diagnostics=SimpleNamespace(add_bytes=lambda *a, **kw: None), generator_free_memory={})
        return kwargs, auditor, state, namespace

    def run_context(self, context):
        kwargs, auditor, state, _ = context
        with patch.dict(os.environ, {c2.MODE_ENV: "timing" if kwargs["args"].timing_mode else "correctness"}), patch.object(c2, "_load_native", return_value=auditor):
            return c2.run_two_cycles(**kwargs)

    def test_actual_helpers_twice_reuse_state_single_construction_and_half_loss(self):
        context = self.context(); result = self.run_context(context); state = context[2]
        self.assertEqual(state.constructs, ["generator", "critic", "ema"])
        self.assertEqual(result["updates"], {"generator": 2, "critic": 10, "ema": 10})
        first, second = result["cycles"]
        self.assertEqual(first["final_versions"], {"generator": 3, "critic": 15, "ema": 15})
        self.assertEqual(second["initial_physical_versions"], first["final_physical_versions"])
        self.assertEqual(second["final_versions"], {"generator": 4, "critic": 20, "ema": 20})
        self.assertEqual(first["object_identity"], second["object_identity"])
        self.assertEqual(state.backward.count(("generator", 2)), 4)
        self.assertEqual(state.backward.count(("critic", 3)), 20)
        self.assertEqual(len([x for x in state.events if isinstance(x, tuple) and x[0] == "rollout"]), 4)
        self.assertEqual(state.pending, set())
        self.assertEqual([x["measured"] for x in result["cycles"]], [False, True])
        self.assertFalse(result["claimable_speedup"])
        self.assertEqual(result["evidence_scope"], "CPU_PROTOCOL_FIXTURE_ONLY")
        self.assertIn("NOT_IMPLEMENTED", result["launcher_auditor_gate"])

    def test_eight_physical_ranks_cover_sixteen_distinct_slots_each_cycle(self):
        rows = [self.run_context(self.context(rank=rank)) for rank in range(8)]
        for cycle in (0, 1):
            self.assertEqual(sorted(slot for row in rows for slot in row["cycles"][cycle]["logical_slots"]), list(range(16)))
            self.assertEqual(sum(row["cycles"][cycle]["census"]["generator_commit"] for row in rows), 8)

    def test_registered_arms_preserve_actual_retention_and_fences(self):
        for arm in ARMS:
            result = self.run_context(self.context(arm=arm))
            for row in result["cycles"]:
                self.assertEqual(row["rollout_fences"]["critic_rollout_sync_count"], 0 if arm == ARMS[2] else 10)
                self.assertEqual(row["generator_nograd_persistent_unshard"]["enabled"], arm != "best")

    def test_missing_native_child_field_fails_closed(self):
        context = self.context(); inner = context[3]["_run_nogan_generator_update_p0"]
        def missing(**kwargs):
            row = inner(**kwargs); del row["rollout"]["empty_cache_before_rollout"]; return row
        context[3]["_run_nogan_generator_update_p0"] = missing
        with self.assertRaises(KeyError): self.run_context(context)

    def test_trace_call_shape_drift_fails_actual_native_auditor(self):
        context = self.context(); inner = context[3]["_run_nogan_generator_update_p0"]
        def wrong(**kwargs):
            row = inner(**kwargs); row["rollout"]["output_shape"] = [1, 16, 9, 24, 40]; return row
        context[3]["_run_nogan_generator_update_p0"] = wrong
        with self.assertRaisesRegex(ValueError, "shape"):
            self.run_context(context)

    def test_early_optimizer_commit_fails_native_microbatch_guard(self):
        context = self.context(); inner = context[3]["_run_nogan_generator_update_p0"]
        def wrong(**kwargs):
            row = inner(**kwargs)
            if not kwargs["expect_optimizer_commit"]: kwargs["generator_engine"].engine.global_steps += 1
            return row
        context[3]["_run_nogan_generator_update_p0"] = wrong
        with self.assertRaisesRegex(AssertionError, "outer boundary"):
            self.run_context(context)

    def test_a100_or_wrong_checkpoint_rejected_before_model_calls(self):
        for mutate in (lambda k: k["result"].update(world_size=16), lambda k: setattr(k["generator_engine"], "global_steps", 3)):
            context = self.context(); mutate(context[0])
            with self.assertRaises(ValueError): self.run_context(context)
            self.assertEqual(context[2].events, [])

    def test_missing_optimizer_execution_is_not_inferred_from_version(self):
        context = self.context(); engine = context[0]["critic_engine"]
        def omit_optimizer():
            engine.lr_scheduler.step(); engine.global_steps += 1
        engine.step = omit_optimizer
        with self.assertRaisesRegex(ValueError, "optimizer/scheduler"):
            self.run_context(context)
        self.assertEqual(len(context[0]["result"]["cycles"]), 1)

    def test_unconsumed_trace_prevents_second_cycle(self):
        context = self.context(); inner = context[3]["_run_nogan_critic_precomputed"]
        def leave_stale(**kwargs):
            row = inner(**kwargs)
            if kwargs["step"] == 14: context[2].pending.add((kwargs["rank"], 14))
            return row
        context[3]["_run_nogan_critic_precomputed"] = leave_stale
        with self.assertRaisesRegex(ValueError, "unconsumed"):
            self.run_context(context)
        self.assertEqual(len(context[0]["result"]["cycles"]), 1)

    def test_correctness_probes_are_after_outer_fence_and_counters_restore(self):
        context = self.context(mode="correctness"); ema = context[0]["ema"]
        original = ema.checksum
        def probe():
            self.assertEqual(context[2].events[-1], "sync")
            self.assertNotIn("step", vars(ema))
            context[2].events.append("probe")
            return original()
        ema.checksum = probe
        result = self.run_context(context)
        self.assertEqual(context[2].events.count("probe"), 2)
        self.assertEqual(context[2].events.count("barrier"), 2)
        self.assertEqual(context[2].events.count("sync"), 4)
        self.assertEqual(result["cycles"][1]["ema_checksum_outside_timer"]["optimization_step"], 20)

    def test_reset_between_cycles_is_rejected_before_second_model_call(self):
        context = self.context(mode="correctness"); original = context[0]["ema"].checksum
        def reset():
            context[0]["generator_engine"].global_steps = 2
            return original()
        context[0]["ema"].checksum = reset
        with self.assertRaisesRegex(ValueError, "continuation"):
            self.run_context(context)
        self.assertEqual(len(context[0]["result"]["cycles"]), 1)

    def test_source_helpers_are_real_unmodified_and_torch_unimported(self):
        imported_before = "torch" in sys.modules
        context = self.context()
        proof = c2.inspect_helper_source(context[0]["p0"])
        self.assertEqual(proof["helper_sha256"], c2.HELPERS)
        self.assertFalse(proof["hook_present"])
        self.assertEqual("torch" in sys.modules, imported_before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
