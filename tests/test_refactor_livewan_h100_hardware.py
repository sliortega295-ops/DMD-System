"""CPU execution checks for physical-eight/logical-sixteen LiveWan routing."""
from __future__ import annotations

from contextlib import ExitStack
from types import SimpleNamespace
import os
import sys
import unittest
from unittest.mock import patch
from pathlib import Path

from distillschedule.runtime.livewan_hardware import livewan_hardware
from distillschedule.runtime import livewan_crossscale_worker as worker


class Tensor:
    def __init__(self, batch=1):
        self.batch = batch

    def contiguous(self):
        return self

    def numel(self):
        return 16

    def element_size(self):
        return 4


class Optimizer:
    def __init__(self):
        self.commits = 0
        self.zeros = 0

    def step(self):
        self.commits += 1

    def zero_grad(self, **_kwargs):
        self.zeros += 1


class Reducer:
    def __init__(self):
        self.commits = 0

    def zero_grad(self):
        pass

    def finish(self):
        self.commits += 1
        return {"group_size": 4}


def row(step, slot):
    return SimpleNamespace(logical_step=step, logical_slot=slot,
        operation_key=f"step={step}/slot={slot}")


class LiveWanH100HardwareTests(unittest.TestCase):
    def test_default_is_original_physical_world16(self):
        with patch.dict(os.environ, {}, clear=True):
            h = livewan_hardware()
        self.assertEqual(h.physical_world_size, 16)
        self.assertEqual(h.generator_pool_size, 8)
        self.assertEqual(h.pipeline_slots(3), (3, 11))
        self.assertEqual(h.critic_accumulations, 1)
        self.assertEqual(h.baseline_slots(15), (15,))

    def test_h100_four_lanes_are_a_logical_batch16_bijection(self):
        h = livewan_hardware("h100-8-b16")
        groups = [h.pipeline_slots(lane) for lane in range(4)]
        self.assertEqual(groups[1], (1, 5, 9, 13))
        self.assertEqual(sorted(sum((list(g) for g in groups), [])), list(range(16)))
        self.assertEqual(h.score_peer(2), 6)
        self.assertEqual(h.score_peer(6), 2)
        self.assertEqual(h.critic_batch_size, 2)
        self.assertEqual(h.critic_accumulations, 2)
        self.assertEqual(h.generator_loss_scale, .25)
        self.assertEqual(h.critic_loss_scale, .5)
        self.assertEqual(h.baseline_slots(3), (3, 11))

    def test_no_unregistered_topology_or_rank(self):
        with self.assertRaises(ValueError):
            livewan_hardware("h100-auto")
        h = livewan_hardware("h100-8-b16")
        for rank in (-1, 8):
            with self.assertRaises(ValueError):
                h.score_peer(rank)
        with self.assertRaises(ValueError):
            h.pipeline_slots(4)

    def test_each_reduction_convention_is_the_same_sixteen_sample_mean(self):
        h = livewan_hardware("h100-8-b16")
        gradients = [i * i - 3 * i + .125 for i in range(16)]
        expected = sum(gradients) / 16
        generator = sum(sum(gradients[s] * h.generator_loss_scale
            for s in h.pipeline_slots(lane)) for lane in range(4)) / 4
        critic = sum(sum(sum(gradients[s] for s in h.pipeline_slots(lane)[i:i+2])
            / 2 * h.critic_loss_scale for i in (0, 2)) for lane in range(4)) / 4
        baseline = sum(sum(gradients[s] / 2 for s in h.baseline_slots(rank))
            for rank in range(8)) / 8
        self.assertEqual((generator, critic, baseline), (expected,) * 3)

    def test_real_registered_c34_plan_is_not_rewritten_into_b32(self):
        from distillschedule.core.config import load_plan
        root = Path(__file__).resolve().parents[1]
        path = root / "configs/distillschedule/plans/livewan_wan13b_14b_sf_dmd_c34_continuous_b16_residual_layout.json"
        with patch.dict(os.environ, {name: "/unread/"+name for name in (
            "LIVEWAN_STUDENT_CHECKPOINT", "LIVEWAN_CRITIC_CHECKPOINT",
            "LIVEWAN_TEACHER_CHECKPOINT", "LIVEWAN_OPERATION_TRACE",
            "LIVEWAN_SYSTEMS_BOOTSTRAP", "DISTILLSCHEDULE_OUTPUT_DIR")}):
            plan = load_plan(path).plan.to_dict()
        original = __import__("copy").deepcopy(plan)
        with patch.dict(os.environ, {"DISTILLSCHEDULE_HARDWARE_PROFILE": "h100-8-b16"}):
            worker._validate_plan_identity(plan, mode="pipeline")
            self.assertEqual(worker._plan_role_ranks(plan, mode="pipeline"),
                (tuple(range(4)), tuple(range(4, 8))))
            self.assertEqual(worker._plan_critic_ranks(plan, mode="pipeline",
                score_ranks=tuple(range(4, 8))), tuple(range(4, 8)))
            self.assertEqual(worker._pipeline_8x8_slot_groups(plan),
                ((0, 4, 8, 12), (1, 5, 9, 13), (2, 6, 10, 14), (3, 7, 11, 15)))
        self.assertEqual(plan, original)
        self.assertEqual(plan["workload"]["shape"]["batch_size"], 16)
        self.assertEqual(worker._source_pipeline_slot_groups(plan),
            worker.TRACE_BALANCED_8G8S_SLOT_PAIRS)

    def test_trajectory_uses_eight_contributors_and_both_logical_rank_rows(self):
        h = livewan_hardware("h100-8-b16")
        seeds = {key: 17 for key in ("rollout", "generator_dmd_t", "generator_dmd_noise",
            "generator_grad_blocks", "critic_t", "critic_noise")}
        def real_row(step, slot):
            return worker.OperationTraceRow(step, slot, slot, "/unread/sample", None,
                f"step={step}/slot={slot}", 3, 0, seeds, step == 0, True)
        global_losses = [{"kind": "critic", "operation_key": real_row(s, i).operation_key,
            "mean": 2.0} for s in range(5) for i in range(16)]
        global_losses += [{"kind": "generator", "operation_key": real_row(0, i).operation_key,
            "mean": 1.0} for i in range(16)]
        contributor_counts = []
        def gather(values, _payload):
            contributor_counts.append(len(values))
            values[:] = [{"losses": global_losses if i == 0 else [],
                "optimizer": [{"role": "generator", "clip_input_norm": 3.0},
                    {"role": "critic", "clip_input_norm": 4.0}],
                "call_census": {"student": 20}} for i in range(8)]
        with patch.dict(os.environ, {"DISTILLSCHEDULE_HARDWARE_PROFILE": h.name}):
            result = worker._trajectory_normalized_cycle_evidence(
                torch=SimpleNamespace(distributed=SimpleNamespace(all_gather_object=gather)),
                rank=3, cycle=0, trace=SimpleNamespace(row=real_row),
                local_losses=[], local_optimizer=[], local_call_census={})
        identities, rng_identities, census, losses, norms, finite = result
        self.assertEqual(contributor_counts, [8])
        self.assertEqual(identities, tuple(f"step={s}/slot={i}" for s in range(5) for i in (3, 11)))
        self.assertEqual(set(rng_identities), set(identities))
        self.assertEqual(census, {"student": 160})
        self.assertEqual(losses, {"generator_mean": 1.0, "critic_mean": 2.0})
        self.assertTrue(finite)

    def test_h100_cannot_reuse_an_a100_timing_admission(self):
        import json
        value = dict(schema_version="distillschedule.livewan_two_cycle_timing_gate.v2",
            status="PASS", claimable_for_two_cycle_timing=True,
            expected_head="a"*40, cycles=2,
            final_versions={"generator": 2, "critic": 10, "ema": 0})
        fake_path = SimpleNamespace(expanduser=lambda: fake_path, resolve=lambda: fake_path,
            is_file=lambda: True, read_text=lambda **_kw: json.dumps(value))
        with patch.dict(os.environ, {"DISTILLSCHEDULE_HARDWARE_PROFILE": "h100-8-b16"}), \
             patch.object(worker, "Path", return_value=fake_path), \
             patch.object(worker, "sha256_file", return_value="b"*64):
            with self.assertRaisesRegex(RuntimeError, "eight-physical-rank correctness"):
                worker._load_two_cycle_timing_gate(path="/unread/a100-gate.json",
                    expected_sha256="b"*64, expected_head="a"*40,
                    mode="pipeline", pipeline_variant="c34", arm_id="c34", seed=17,
                    trace=None, bootstrap=None, source_receipt={})

    def _dependencies(self, stack):
        fake_dist = SimpleNamespace(recv=lambda *_a, **_k: None,
            send=lambda *_a, **_k: None,
            isend=lambda *_a, **_k: SimpleNamespace(wait=lambda: None))
        fake_torch = SimpleNamespace(distributed=fake_dist,
            empty_like=lambda value: Tensor(value.batch),
            cat=lambda values, dim: Tensor(sum(x.batch for x in values)),
            nn=SimpleNamespace(utils=SimpleNamespace(clip_grad_norm_=lambda *_a, **_k: 0)))
        stack.enter_context(patch.dict(sys.modules, {"torch": fake_torch,
            "torch.distributed": fake_dist}))
        stack.enter_context(patch.object(worker, "_record_output", lambda *_a, **_k: None))
        stack.enter_context(patch.object(worker, "_record_optimizer_diagnostics", lambda *_a, **_k: None))
        stack.enter_context(patch.object(worker, "prepare_score_sample_context",
            lambda *_a, **_k: SimpleNamespace(critic=Tensor(), teacher=Tensor())))
        stack.enter_context(patch.object(worker, "_recv_pair", lambda *_a, **_k: (Tensor(), None)))
        stack.enter_context(patch.object(worker, "_send_pair", lambda **_k: ()))
        stack.enter_context(patch.object(worker, "_dmd_flow_inputs", lambda *_a, **_k: None))
        stack.enter_context(patch.object(worker, "_dmd_fake_velocity", lambda *_a, **_k: Tensor()))
        stack.enter_context(patch.object(worker, "_dmd_teacher_velocity", lambda *_a, **_k: Tensor()))
        stack.enter_context(patch.object(worker, "_combine_dmd_gradient", lambda *_a, **_k: Tensor()))
        return fake_torch

    def test_actual_four_wave_runner_critic_is_two_b2_backwards_then_one_commit(self):
        calls = []
        h = livewan_hardware("h100-8-b16")
        reducer = Reducer()
        opt = Optimizer()
        model = SimpleNamespace(parameters=lambda: [])
        bundle = SimpleNamespace(critic=model, teacher=model,
            critic_gradient_reducer=reducer, critic_lora_params=[], critic_optimizer=opt)
        trace = SimpleNamespace(global_batch=16, row=row)
        transport = SimpleNamespace(waves_per_step=4, receipt=lambda: {},
            receive_to_device=lambda **_k: Tensor())
        sync = SimpleNamespace(wait_source=lambda **_k: None)
        loader = SimpleNamespace(context_from_trace=lambda *_a, **_k: Tensor())
        def backward(_torch, **kwargs):
            calls.append((tuple(r.logical_slot for r in kwargs["rows"]),
                kwargs["x0"].batch, kwargs["loss_scale"], kwargs["reducer_accumulation"], opt.commits))
            return [0, 0]
        versions = {"generator": 0, "critic": 0, "ema": 0}
        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, {"DISTILLSCHEDULE_HARDWARE_PROFILE": h.name}))
            torch = self._dependencies(stack)
            stack.enter_context(patch.object(worker, "critic_backward_for_batch", backward))
            worker._run_pipeline_8x8_b32(torch=torch, rank=5, trace=trace,
                loader=loader, bundle=bundle, streamers=[], score_context=object(),
                run_kind="timing", evidence=worker.RunEvidence(), versions=versions,
                device=None, slot_groups_by_lane=tuple(h.pipeline_slots(i) for i in range(4)),
                runahead_host_transport=transport, critic_shadow_hot_sync=sync,
                step_offset=0, cycle_index=0, precomputed_first=None)
        self.assertEqual(len(calls), 10)
        for step in range(5):
            a, b = calls[step*2:step*2+2]
            self.assertEqual(a, ((1, 5), 2, .5, (0, 2), step))
            self.assertEqual(b, ((9, 13), 2, .5, (1, 2), step))
        self.assertEqual(opt.commits, 5)
        self.assertEqual(reducer.commits, 5)
        self.assertEqual(versions, {"generator": 0, "critic": 5, "ema": 0})

    def test_actual_four_wave_trajectory_retains_all_twenty_critic_loss_identities(self):
        h = livewan_hardware("h100-8-b16")
        model = SimpleNamespace(parameters=lambda: [])
        bundle = SimpleNamespace(critic=model, teacher=model,
            critic_gradient_reducer=Reducer(), critic_lora_params=[], critic_optimizer=Optimizer())
        transport = SimpleNamespace(waves_per_step=4, receipt=lambda: {}, receive_to_device=lambda **_k: Tensor())
        evidence = worker.RunEvidence()
        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, {"DISTILLSCHEDULE_HARDWARE_PROFILE": h.name}))
            torch = self._dependencies(stack)
            stack.enter_context(patch.object(worker, "critic_backward_for_batch", lambda *_a, **_kw: [0,0]))
            stack.enter_context(patch.object(worker, "_tensor_fingerprint", lambda *_a: {"mean": 1.0}))
            worker._run_pipeline_8x8_b32(torch=torch, rank=5, trace=SimpleNamespace(global_batch=16,row=row),
                loader=SimpleNamespace(context_from_trace=lambda *_a, **_k: Tensor()), bundle=bundle,
                streamers=[], score_context=object(), run_kind="trajectory", evidence=evidence,
                versions={"generator":0,"critic":0,"ema":0}, device=None,
                slot_groups_by_lane=tuple(h.pipeline_slots(i) for i in range(4)),
                runahead_host_transport=transport, critic_shadow_hot_sync=SimpleNamespace(wait_source=lambda **_k: None),
                step_offset=0, cycle_index=0, precomputed_first=None)
        self.assertEqual(len(evidence.loss_fingerprints),20)
        self.assertEqual({r["operation_key"] for r in evidence.loss_fingerprints},
            {f"step={step}/slot={slot}" for step in range(5) for slot in (1,5,9,13)})

    def test_actual_four_wave_runner_generator_commits_once_after_four_b1_losses(self):
        calls, prepared_rows, staged = [], [], []
        h = livewan_hardware("h100-8-b16")
        opt = Optimizer()
        model = SimpleNamespace(parameters=lambda: [])
        bundle = SimpleNamespace(student=model, critic_shadow=model,
            generator_optimizer=opt, generator_group=None, generator_group_size=4)
        trace = SimpleNamespace(global_batch=16, row=row)
        transport = SimpleNamespace(waves_per_step=4, receipt=lambda: {},
            enqueue_send=lambda **kw: staged.append((kw["step"], kw["wave"])),
            wait_staged=lambda **_k: None)
        sync = SimpleNamespace(receive_and_load=lambda **_k: None)
        def prepare(_torch, **kw):
            prepared_rows.append((kw["row"].logical_step, kw["row"].logical_slot, opt.commits))
            return SimpleNamespace(row=kw["row"], x0=Tensor(), raw_context=Tensor(), context_lens=None)
        def backward(_torch, **kw):
            calls.append((kw["prepared"].row.logical_slot, kw["sample_weight"], opt.commits))
            return []
        versions = {"generator": 0, "critic": 0, "ema": 0}
        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, {"DISTILLSCHEDULE_HARDWARE_PROFILE": h.name}))
            torch = self._dependencies(stack)
            stack.enter_context(patch.object(worker, "prepare_rollout", prepare))
            stack.enter_context(patch.object(worker, "generator_backward_for_sample", backward))
            stack.enter_context(patch.object(worker, "_mean_replicated_gradients", lambda *_a, **kw: {"group_size": kw["group_size"]}))
            worker._run_pipeline_8x8_b32(torch=torch, rank=2, trace=trace,
                loader=object(), bundle=bundle, streamers=[object(), object()], score_context=object(),
                run_kind="timing", evidence=worker.RunEvidence(), versions=versions,
                device=None, slot_groups_by_lane=tuple(h.pipeline_slots(i) for i in range(4)),
                runahead_host_transport=transport, critic_shadow_hot_sync=sync,
                step_offset=0, cycle_index=0, precomputed_first=None)
        self.assertEqual(calls, [(2, .25, 0), (6, .25, 0), (10, .25, 0), (14, .25, 0)])
        self.assertEqual(opt.commits, 1)
        self.assertEqual(len(prepared_rows), 20)
        self.assertTrue(all(commits == 1 for step, _slot, commits in prepared_rows if step > 0))
        self.assertEqual(staged, [(step, wave) for step in range(1, 5) for wave in range(4)])

    def test_actual_baseline_two_b1_rows_are_accumulated_before_each_commit(self):
        h = livewan_hardware("h100-8-b16")
        gopt, copt = Optimizer(), Optimizer()
        g_calls, c_calls, prepared_rows = [], [], []
        model = SimpleNamespace(parameters=lambda: [])
        bundle = SimpleNamespace(student=model, critic=model, teacher=model,
            generator_optimizer=gopt, critic_optimizer=copt, critic_lora_params=[],
            score_group=None, score_group_size=8)
        def prepare(_torch, **kw):
            r = kw["row"]
            prepared_rows.append((r.logical_step, r.logical_slot, gopt.commits, copt.commits))
            return SimpleNamespace(row=r, x0=Tensor(), raw_context=Tensor(), context_lens=None)
        def g_backward(_torch, **kw):
            g_calls.append((kw["prepared"].row.logical_slot, kw["sample_weight"], gopt.commits))
            return []
        def c_backward(_torch, **kw):
            c_calls.append((kw["row"].logical_step, kw["row"].logical_slot,
                kw["sample_weight"], copt.commits))
            return 0
        versions = {"generator": 0, "critic": 0, "ema": 0}
        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, {"DISTILLSCHEDULE_HARDWARE_PROFILE": h.name}))
            torch = self._dependencies(stack)
            stack.enter_context(patch.object(worker, "prepare_rollout", prepare))
            stack.enter_context(patch.object(worker, "dmd_gradient_for_sample", lambda *_a, **_k: Tensor()))
            stack.enter_context(patch.object(worker, "generator_backward_for_sample", g_backward))
            stack.enter_context(patch.object(worker, "critic_backward_for_sample", c_backward))
            stack.enter_context(patch.object(worker, "_mean_replicated_gradients", lambda *_a, **kw: {"group_size": kw["group_size"]}))
            worker._run_baseline(torch=torch, rank=3, trace=SimpleNamespace(row=row),
                loader=object(), bundle=bundle, streamers=[object()], score_context=object(),
                run_kind="timing", evidence=worker.RunEvidence(), versions=versions, device=None)
        self.assertEqual(g_calls, [(3, .5, 0), (11, .5, 0)])
        self.assertEqual(len(c_calls), 10)
        for step in range(5):
            self.assertEqual(c_calls[step*2:step*2+2], [(step, 3, .5, step), (step, 11, .5, step)])
        self.assertEqual((gopt.commits, copt.commits), (1, 5))
        self.assertEqual(versions, {"generator": 1, "critic": 5, "ema": 0})


if __name__ == "__main__":
    unittest.main()
