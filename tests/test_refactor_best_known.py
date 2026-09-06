from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from distillschedule.workloads.best_known.registry import (
    CANONICAL_IDS,
    GPU_NOT_RUN,
    fingerprint,
    list_profiles,
    load_profile,
    source_layout,
)

EXPECTED = {
    "wan13b_nogan_resident_best",
    "helios14b_nogan_best",
    "minimax_h3_grid1000_best",
    "helios14b_gan_best",
    "livewan_crossscale_pipeline_best",
}


class BestKnownIntegrationTests(unittest.TestCase):
    def test_registry_exactly_five(self):
        self.assertEqual(set(CANONICAL_IDS), EXPECTED)
        self.assertEqual({p["id"] for p in list_profiles(root=ROOT)}, EXPECTED)

    def test_lock_hashes_are_stable_and_no_unresolved_defaults(self):
        lock = json.loads((ROOT / "configs/distillschedule/best_known/registry.lock.json").read_text())["profiles"]
        self.assertEqual(set(lock), EXPECTED)
        for wid in EXPECTED:
            p = load_profile(wid, root=ROOT)
            self.assertEqual(lock[wid], fingerprint(p))
            raw = json.dumps(p, sort_keys=True)
            self.assertNotIn("${", raw)
            self.assertNotIn("UNRESOLVED", raw)

    def test_canonical_mutation_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            shutil.copytree(ROOT / "configs", root / "configs")
            # repository_root-dependent source files are irrelevant: lock failure happens first.
            p = root / "configs/distillschedule/best_known/minimax_h3_grid1000_best.json"
            value = json.loads(p.read_text())
            value["runtime"]["environment"]["H3_ACTIVATION_POLICY"] = "checkpoint_boundary_cpu"
            p.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "canonical profile changed"):
                load_profile("minimax_h3_grid1000_best", root=root)

    def test_wan_winner_flags(self):
        p = load_profile("wan13b_nogan_resident_best", root=ROOT)
        self.assertEqual(p["topology"]["placement"], "resident-async128-tree-chunk2048")
        self.assertEqual(p["runtime"]["environment"]["DISTILLSCHEDULE_WAN13B_GENERATOR_NOGRAD_LORA_FUSED_EPILOGUE"], "1")
        self.assertEqual(p["contract"]["actual_global_batch"], 16)

    def test_helios_nogan_winner_flags(self):
        p = load_profile("helios14b_nogan_best", root=ROOT)
        self.assertEqual(p["contract"]["variant"], "no-gan")
        self.assertEqual(p["topology"], {"placement": "node-hsdp", "replicate": 2, "rollout_batch_size": 1, "shard": 8})
        env = p["runtime"]["environment"]
        self.assertEqual(env["DISTILLSCHEDULE_HELIOS_ATTENTION_BACKEND"], "local-fa2")
        self.assertEqual(env["DISTILLSCHEDULE_HELIOS_SELECTIVE_FLASH_CHECKPOINT_STRIDE"], "2")
        self.assertEqual(env["DISTILLSCHEDULE_OPTIMIZER_DISPATCH_WARMUP"], "on")

    def test_h3_exact_winner_vector_and_approximate_semantics(self):
        p = load_profile("minimax_h3_grid1000_best", root=ROOT)
        env = p["runtime"]["environment"]
        expected = {
            "H3_ATTN_BACKEND": "_flash_3_hub", "H3_BENCHMARK_SEED": "20260817",
            "H3_ACTIVATION_CHECKPOINT_SEGMENT": "1", "H3_ACTIVATION_POLICY": "none",
            "H3_FA3_NOGRAD_NUM_SPLITS": "2", "H3_FA3_GRAD_NUM_SPLITS": "2",
            "H3_FA3_REPLAY_CACHE_BLOCKS": "0-49", "H3_FA3_REPLAY_CACHE_STORAGE": "cpu_staged",
            "H3_FA3_REPLAY_CACHE_MAX_D2H_INFLIGHT": "2", "H3_FA3_REPLAY_CACHE_TRIM_BEFORE_BACKWARD": "1",
            "H3_FUSED_BLOCK_POINTWISE": "1", "H3_FUSED_BLOCK_POINTWISE_GRAD": "1",
            "H3_FUSED_RMSNORM_MODULATE": "1", "H3_FUSED_ROTARY": "1", "H3_FUSED_ROTARY_GRAD": "1",
            "H3_FUSED_QK_RMSNORM_ROTARY": "1", "H3_FUSED_QK_RMSNORM_ROTARY_GRAD": "1",
            "H3_FUSED_SWIGLU": "1", "H3_FUSED_SWIGLU_GRAD": "1",
            "H3_LORA_SCALE1_ELISION": "1", "H3_LORA_NOGRAD_EPILOGUE": "1", "H3_LORA_GRAD_EPILOGUE": "1",
            "H3_LORA_EPILOGUE_BLOCK_N": "128",
        }
        for k, v in expected.items(): self.assertEqual(env[k], v, k)
        self.assertEqual(p["semantics"], "approximate_grid1000")
        self.assertFalse(p["memory"]["activation_boundary_offload"])
        self.assertEqual(p["memory"]["fa3_replay_storage"], "cpu_staged")
        self.assertTrue((ROOT / "src/distillschedule/backends/training/minimax_h3/trainer.py").is_file())
        self.assertFalse(any(u["url"].endswith("H3-DMD-for-A100.git") for u in p["upstreams"]))

    def test_gan_is_c25_not_later_canary(self):
        p = load_profile("helios14b_gan_best", root=ROOT)
        self.assertEqual(p["contract"]["logical_steps"], [1000,1001,1002,1003,1004])
        self.assertEqual(p["contract"]["updates"], {"generator":1,"critic":5,"ema":1})
        self.assertEqual(p["contract"]["final_versions"], {"generator":201,"critic":1005,"ema":801})
        self.assertEqual(p["contract"]["winner"], "C25")
        self.assertEqual(p["runtime"]["environment"]["DISTILLSCHEDULE_HELIOS_EXACT_HISTORY_CURRENT_FUSION"], "on")
        self.assertNotIn("C27", json.dumps(p))

    def test_livewan_is_actual_b16_c24(self):
        p = load_profile("livewan_crossscale_pipeline_best", root=ROOT)
        self.assertEqual(p["contract"]["actual_global_batch"], 16)
        self.assertFalse(p["contract"]["timing_normalized_from_B32"])
        self.assertEqual(p["historical"]["seconds"], 54.441603646)
        self.assertNotEqual(p["historical"]["seconds"], 54.291817154)
        self.assertEqual(p["runtime"]["environment"]["LIVEWAN_PIPELINE_VARIANT"], "c24-continuous-b16-score-modulation")
        self.assertTrue(p["runtime"]["typed_plan"].endswith("c24_continuous_b16_score_modulation.json"))

    def test_claim_boundary_is_explicit(self):
        for p in list_profiles(root=ROOT):
            self.assertTrue(p["validation"]["historical_source_validated"])
            self.assertFalse(p["validation"]["integration_ci_validated"])
            self.assertEqual(p["validation"]["server_revalidation"], GPU_NOT_RUN)
            self.assertFalse(p["historical"]["formal_speedup_eligible"])

    def test_source_layout_and_gitlink_pins(self):
        for p in list_profiles(root=ROOT):
            self.assertEqual(source_layout(ROOT, p)["status"], "PASS")

    def test_no_new_site_specific_absolute_paths(self):
        roots = [
            ROOT / "configs/distillschedule/best_known",
            ROOT / "src/distillschedule/workloads/best_known",
            ROOT / "src/distillschedule/backends/training/minimax_h3",
        ]
        files = []
        for base in roots:
            files.extend(p for p in base.rglob("*") if p.is_file())
        files += [
            ROOT / "scripts/distillschedule/launch_wan13b_fair_world16_node.sh",
            ROOT / "scripts/distillschedule/launch_static_fsdp2_p0_world16_node.sh",
            ROOT / "scripts/distillschedule/launch_livewan_crossscale_world16_node.sh",
            ROOT / "scripts/distillschedule/launch_minimax_h3_best_node.sh",
        ]
        bad = []
        for p in files:
            text = p.read_text(errors="ignore")
            if "/share/project/" in text or "/home/shuchenweng/" in text or "/home/lyy/" in text:
                bad.append(str(p.relative_to(ROOT)))
        self.assertEqual(bad, [])

    def test_no_large_experiment_binaries_added(self):
        forbidden = {".nsys-rep", ".sqlite", ".pt", ".pth", ".ckpt", ".safetensors"}
        bad=[]
        for base in [ROOT / "configs/distillschedule/best_known", ROOT / "src/distillschedule/backends/training/minimax_h3"]:
            for p in base.rglob("*"):
                if p.is_file() and (p.suffix in forbidden or p.stat().st_size > 2_000_000): bad.append(str(p))
        self.assertEqual(bad, [])


if __name__ == "__main__":
    unittest.main()
