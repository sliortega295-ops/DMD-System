"""Stdlib-only controls; fixture PASS is never native GPU evidence.

Set DMD_HELIOS_H100_NATIVE_ROOT to the clean registered 8fc70e8 checkout for
the additional real-source compiler test. It imports no Torch and runs no GPU.
"""
import copy
import os
from pathlib import Path
import sys
import subprocess
import unittest
from unittest.mock import patch

from distillschedule.search import hardware_helios as h
from distillschedule.search.allocation import ExecutionAllocation, PhysicalDevice


def deployment():
    slots = tuple((r, r + 8) for r in range(8))
    allocation = ExecutionAllocation(tuple(PhysicalDevice(r, 0, r, f"GPU-fixture-{r}", "sm90", 80 * 2**30)
        for r in range(8)), 16, {"generator": slots, "critic": slots})
    return {"schema_version": "distillschedule.hardware_helios_deployment.v1",
        "native_source": {"head": h.NATIVE_HEAD, "repo_root": "/fixture/native"},
        "site": {"project_root": "/fixture/project", "python": "/fixture/python/bin/python",
                 "helios_root": "/fixture/helios", "asset_root": "/fixture/assets", "hf_home": "/fixture/assets/hf"},
        "allocation": allocation.to_dict(), "asset_gate": {"path": "/fixture/asset-gate.json", "sha256": "a" * 64},
        "retention_api": {"path": "/fixture/api.json", "sha256": h.RETENTION_SHA},
        "runtime_versions": {"python": "3.10.20", "driver": "fixture-driver",
            "packages": dict(torch="2.10.0+cu128", triton="3.6.0", deepspeed="0.18.4", accelerate="1.12.0",
                             transformers="5.3.0", diffusers="0.38.0.dev0", peft="0.18.1", safetensors="0.6.2")},
        "master_port": 28791, "deadline_seconds": 1800}


def native_fixture(root, action, payload):
    assert action == "inventory"
    return {arm: {"frozen": {"arm": arm, "fixture_scope": "CPU_FIXTURE_ONLY"},
                  "policy_argv": ["--attention-backend", "local-fa2"],
                  "workload_contract": {"resolution": [384, 640]}} for arm in payload["arms"]}


def mechanism_fixture(arm):
    retained, nosync = arm != "best", arm == "h100-nograd-retain-nosync"
    rows = []
    for rank in range(8):
        p = {"enabled": retained, "rollout_batch_count": 5}
        if retained:
            p.update(fsdp_module_count=41, set_no_reshard_count=41, restore_reshard_policy_count=41,
                     explicit_reshard_count=41, fully_unsharded_before_exit_count=41,
                     unsharded_after_exit_count=0, restored=True, flush_allocator_cache=True,
                     pre_empty_cache_count=1, post_empty_cache_count=1)
        rows.append({"rank": rank, "fixture_scope": "CPU_FIXTURE_ONLY",
            "attention_backend": {"status": "PASS", "requested": "local-fa2", "runtime": "local-fa2", "extension_sha256": h.FLASH_SHA},
            "generator_nograd_persistent_unshard": {"logical_b1_microbatch_count": 2, "rollout_batch_count": 10,
                "microbatches": [{"logical_slot": slot, "persistence": copy.deepcopy(p)} for slot in (rank, rank + 8)]},
            "rollout_fence_policy": {"elide_explicit_rollout_sync": nosync,
                "generator_update_empty_cache_count": 0 if nosync else 2, "generator_update_sync_count": 0 if nosync else 2,
                "critic_rollout_sync_count": 0 if nosync else 10, "outer_full5_boundary_sync_preserved": True}})
    return rows


class HardwareHeliosTests(unittest.TestCase):
    def compile(self, **kwargs):
        args = dict(arm="h100-nograd-retain-nosync", mode="correctness", run_id="cpu-fixture")
        args.update(kwargs)
        with patch.object(h, "_native_control", side_effect=native_fixture):
            return h.compile_native_plan(deployment(), **args)

    def test_argv_is_exact_existing_eight_argument_launcher(self):
        p = self.compile()
        self.assertEqual(p["argv"], ["bash", "/fixture/native/" + h.LAUNCHER, "/fixture/project", "cpu-fixture",
            "no-gan", "correctness", "h100-nograd-retain-nosync", h.NATIVE_HEAD, "/fixture/asset-gate.json", "a" * 64])
        self.assertEqual(p["environment"]["DISTILLSCHEDULE_HELIOS_H100_RETENTION_API_RECEIPT"], "/fixture/api.json")

    def test_existing_full5_cannot_be_reported_as_c2(self):
        p = self.compile()
        self.assertEqual(p["timer"]["continuous_cycles"], 1)
        self.assertFalse(p["timer"]["claimable_cycle2"])
        self.assertFalse(p["claimable_speedup"])
        self.assertEqual(p["execution_status"], "NOT_RUN")
        self.assertIn("CAMPAIGN_STOP", p["authorization"])

    def test_timing_requires_exact_correctness_reference(self):
        with self.assertRaisesRegex(ValueError, "correctness"):
            self.compile(mode="timing")
        p = self.compile(mode="timing", correctness={"path": "/fixture/correctness.json", "sha256": "b" * 64})
        self.assertEqual(p["environment"]["DISTILLSCHEDULE_HELIOS_H100_CORRECTNESS_SHA256"], "b" * 64)

    def test_correctness_cannot_inherit_timing_reference(self):
        with self.assertRaisesRegex(ValueError, "non-timing"):
            self.compile(correctness={"path": "/fixture/correctness.json", "sha256": "b" * 64})

    def test_reference_arm_does_not_enable_retention(self):
        self.assertNotIn("DISTILLSCHEDULE_HELIOS_H100_RETENTION_API_RECEIPT", self.compile(arm="best")["environment"])

    def test_forbidden_candidate_and_mode(self):
        for kw in ({"arm": "fa3"}, {"mode": "c2"}, {"mode": "calibration"}, {"run_id": "../escape"}):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                self.compile(**kw)

    def test_no_freeform_environment_or_semantic_override(self):
        for key, value in (("environment", {"ATTENTION": "fa3"}), ("resolution", [128, 128]), ("cycles", 2)):
            d = deployment(); d[key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "unknown/missing"):
                h.validate_deployment(d)

    def test_reject_a100_duplicate_slots_and_physical_b2(self):
        for mutation in (lambda d: d["allocation"]["devices"][0].update(architecture="sm80"),
                         lambda d: d["allocation"]["role_slots"]["generator"][0].__setitem__(1, 0),
                         lambda d: d["allocation"].update(physical_microbatch=2)):
            d = deployment(); mutation(d)
            with self.assertRaises(ValueError):
                h.validate_deployment(d)

    def test_reject_unregistered_source_and_api_proof(self):
        for section, key in (("native_source", "head"), ("retention_api", "sha256")):
            d = deployment(); d[section][key] = "0" * len(d[section][key])
            with self.assertRaises(ValueError):
                h.validate_deployment(d)

    def test_plan_tampering_is_rejected(self):
        for key, value in (("arm", "best"), ("argv", []), ("timer", {"claimable_cycle2": True}), ("claimable_speedup", True)):
            p = self.compile(); p[key] = value
            with patch.object(h, "_native_control", side_effect=native_fixture), self.assertRaisesRegex(ValueError, "altered"):
                h.validate_native_plan(p)

    def test_all_three_mechanism_fixtures(self):
        for arm in h.ARMS:
            self.assertEqual(h.audit_mechanisms(mechanism_fixture(arm), arm=arm)["physical_ranks"], 8)

    def test_fa3_cannot_masquerade_as_fa2(self):
        rows = mechanism_fixture(h.ARMS[2]); rows[0]["attention_backend"]["runtime"] = "fa3"
        with self.assertRaisesRegex(ValueError, "FA3"):
            h.audit_mechanisms(rows, arm=h.ARMS[2])

    def test_missing_rank_and_slot_alias_are_rejected(self):
        rows = mechanism_fixture(h.ARMS[2])
        with self.assertRaises(ValueError):
            h.audit_mechanisms(rows[:-1], arm=h.ARMS[2])
        rows[0]["generator_nograd_persistent_unshard"]["microbatches"][1]["logical_slot"] = 0
        with self.assertRaisesRegex(ValueError, "slots"):
            h.audit_mechanisms(rows, arm=h.ARMS[2])

    def test_retention_restore_and_allocator_fences_are_required(self):
        for key, value in (("restored", False), ("unsharded_after_exit_count", 41), ("post_empty_cache_count", 0), ("fsdp_module_count", 40)):
            rows = mechanism_fixture(h.ARMS[2]); rows[0]["generator_nograd_persistent_unshard"]["microbatches"][1]["persistence"][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "restored"):
                h.audit_mechanisms(rows, arm=h.ARMS[2])

    def test_nosync_must_execute_and_outer_fences_must_survive(self):
        for key, value in (("critic_rollout_sync_count", 10), ("critic_rollout_sync_count", False),
                           ("outer_full5_boundary_sync_preserved", False), ("outer_full5_boundary_sync_preserved", 1)):
            rows = mechanism_fixture(h.ARMS[2]); rows[0]["rollout_fence_policy"][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "fence"):
                h.audit_mechanisms(rows, arm=h.ARMS[2])

    def test_no_gpu_execution_api_is_present(self):
        self.assertFalse(hasattr(h, "run_selected"))
        self.assertFalse(hasattr(h, "calibrate"))
        # Other tests may already import Torch during suite collection. Prove
        # this module's import isolation in a fresh interpreter instead.
        subprocess.run([sys.executable, "-c",
            "import sys; from distillschedule.search import hardware_helios; "
            "assert 'torch' not in sys.modules"], check=True)


@unittest.skipUnless(os.environ.get("DMD_HELIOS_H100_NATIVE_ROOT"), "explicit local registered source needed")
class HardwareHeliosNativeSourceTests(unittest.TestCase):
    def test_real_source_policies_compile_without_torch(self):
        root = Path(os.environ["DMD_HELIOS_H100_NATIVE_ROOT"])
        for arm in h.ARMS:
            p = h.compile_native_plan(deployment(), arm=arm, mode="correctness", run_id="source-only-" + arm, source_root=root)
            policy = p["native_policy"]["frozen"]["policy"]
            self.assertEqual(p["native_policy"]["frozen"]["mesh_shape"], [2, 4])
            self.assertEqual(p["native_policy"]["workload_contract"]["resolution"], [384, 640])
            self.assertEqual(policy["attention_backend"], "local-fa2")
            self.assertEqual(policy["selective_flash_checkpoint_stride"], 2)
            self.assertEqual(policy["generator_nograd_persistent_unshard"], arm != "best")
            self.assertEqual(policy["elide_explicit_rollout_sync"], arm == h.ARMS[2])
            self.assertIn("--attention-backend", p["native_policy"]["policy_argv"])
        # The source inspector itself runs in a fresh interpreter; a parent
        # pytest process is allowed to have imported Torch in another test.


if __name__ == "__main__":
    unittest.main(verbosity=2)
