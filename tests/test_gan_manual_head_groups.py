"""CPU-only characterization of the opt-in, source-pinned GAN head grouping.

Torch, FSDP and Helios are metadata fakes; these tests do not claim GPU or
numerical parity. The real owner and alias checks are exercised unchanged.
"""
from __future__ import annotations

import hashlib
import importlib
import inspect
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from distillschedule.runtime import fsdp2_p0 as policy


class Parameter:
    pass


class Block:
    def __init__(self):
        self.weights = [("weight", Parameter())]

    def named_parameters(self, *, remove_duplicate=True):
        return iter(self.weights)


class Head(Block):
    pass


class ModuleDict(dict):
    pass


class Model:
    def __init__(self, count=40):
        self.blocks = [Block() for _ in range(count)]
        self.is_use_gan = True
        self.gan_heads = ModuleDict((key, Head()) for key in ("5", "15", "25", "35"))
        self.gan_final_head = Head()
        self.extra_modules = []
        self.extra_parameters = []

    def named_modules(self, *, remove_duplicate=True):
        return iter([
            ("", self),
            *((f"blocks.{i}", block) for i, block in enumerate(self.blocks)),
            ("gan_heads", self.gan_heads),
            *((f"gan_heads.{key}", value) for key, value in self.gan_heads.items()),
            ("gan_final_head", self.gan_final_head),
            *self.extra_modules,
        ])

    def named_parameters(self, *, remove_duplicate=True):
        rows = list(self.extra_parameters)
        for name, child in self.named_modules(remove_duplicate=False):
            if child is not self and hasattr(child, "named_parameters"):
                rows.extend((name + "." + key, value) for key, value in child.named_parameters(remove_duplicate=False))
        if remove_duplicate:
            seen = set()
            rows = [(name, value) for name, value in rows if not (id(value) in seen or seen.add(id(value)))]
        return iter(rows)

    def parameters(self):
        return (value for _, value in self.named_parameters())


class HeadGroupPortTests(unittest.TestCase):
    def setUp(self):
        self.shards = []
        self.mesh = object()
        torch = ModuleType("torch")
        dist = ModuleType("torch.distributed")
        fsdp = ModuleType("torch.distributed.fsdp")
        transformer = ModuleType("helios.modules.transformer_helios")
        torch.bfloat16, torch.float32 = "torch.bfloat16", "torch.float32"
        torch.nn = SimpleNamespace(ModuleDict=ModuleDict)
        torch.cuda = SimpleNamespace(synchronize=lambda: None)
        torch.distributed = dist
        dist.is_initialized = lambda: True
        dist.get_world_size = lambda: 16
        fsdp.MixedPrecisionPolicy = lambda **kw: SimpleNamespace(**kw)

        def fully_shard(module, **kw):
            self.shards.append((module, kw))
            for _, parameter in module.named_parameters():
                parameter.device_mesh = self.mesh

        fsdp.fully_shard = fully_shard
        transformer.Discriminator3DHead = Head
        self.addCleanup(patch.stopall)
        patch.dict(sys.modules, {
            "torch": torch,
            "torch.distributed": dist,
            "torch.distributed.fsdp": fsdp,
            "helios": ModuleType("helios"),
            "helios.modules": ModuleType("helios.modules"),
            "helios.modules.transformer_helios": transformer,
        }).start()
        patch.object(policy, "create_device_mesh", lambda *a, **kw: self.mesh).start()

    @staticmethod
    def legacy_receipt(count, mesh):
        return {
            "backend": "FSDP2_COMPOSABLE_FULL_SHARD" if mesh == "world16-fsdp" else "FSDP2_COMPOSABLE_HYBRID_SHARD",
            "mesh_policy": mesh, "mesh_shape": [16] if mesh == "world16-fsdp" else [2, 8],
            "world_size": 16, "block_count": count,
            "wrap_policy": f"{count}_BLOCKS_BOTTOM_UP_THEN_ROOT",
            "reshard_after_forward": True, "param_compute_dtype": "torch.bfloat16",
            "reduce_dtype": "torch.float32", "cpu_parameter_offload": False,
            "parameter_streaming": False, "transition": False,
        }

    def test_legacy_default_receipt_characterization_including_wan30(self):
        for count, mesh in ((40, "world16-fsdp"), (40, "node-hsdp"), (30, "world16-fsdp")):
            with self.subTest(count=count, mesh=mesh):
                model = Model(count)
                self.shards.clear()
                actual = policy.apply_blockwise_fsdp2_policy(model, mesh_policy=mesh, expected_block_count=count)
                self.assertEqual(json.dumps(actual, sort_keys=True), json.dumps(self.legacy_receipt(count, mesh), sort_keys=True))
                self.assertEqual([m for m, _ in self.shards], [*model.blocks, model])

    def test_new_option_is_keyword_only_default_false(self):
        arg = inspect.signature(policy.apply_blockwise_fsdp2_policy).parameters["separate_gan_heads"]
        self.assertIs(arg.default, False)
        self.assertEqual(arg.kind, inspect.Parameter.KEYWORD_ONLY)

    def test_off_does_not_add_receipt_fields_or_change_wan30(self):
        for count in (30, 40):
            model = Model(count)
            actual = policy.apply_blockwise_fsdp2_policy(model, mesh_policy="world16-fsdp", expected_block_count=count, separate_gan_heads=False)
            self.assertEqual(actual, self.legacy_receipt(count, "world16-fsdp"))

    def test_original_helper_bytes_are_pinned(self):
        helper = importlib.import_module("distillschedule.runtime.fsdp2_gan_head_groups")
        self.assertEqual(hashlib.sha256(Path(helper.__file__).read_bytes()).hexdigest(),
                         "085487dd4c38a5ccfee57f91ed56bc82f20ae9dcfeb7dcec83af9dbb0f564aff")

    def test_enabled_wraps_40_blocks_then_5_owned_heads_then_root(self):
        model = Model()
        receipt = policy.apply_blockwise_fsdp2_policy(model, mesh_policy="world16-fsdp", separate_gan_heads=True)
        heads = [*model.gan_heads.values(), model.gan_final_head]
        self.assertEqual([m for m, _ in self.shards], [*model.blocks, *heads, model])
        self.assertEqual(receipt["wrap_policy"], "40_BLOCKS_THEN_5_GAN_HEADS_THEN_ROOT")
        self.assertEqual(receipt["separate_gan_head_fsdp"], {
            "requested": True, "enabled": True,
            "head_paths": ["gan_heads.5", "gan_heads.15", "gan_heads.25", "gan_heads.35", "gan_final_head"],
            "head_unit_count": 5, "fsdp_unit_count": 46, "original_gradient_mesh_size": 16,
        })
        for _, kwargs in self.shards:
            self.assertIs(kwargs["mesh"], self.mesh)
            self.assertIs(kwargs["reshard_after_forward"], True)
            self.assertEqual(kwargs["mp_policy"].param_dtype, "torch.bfloat16")
            self.assertEqual(kwargs["mp_policy"].reduce_dtype, "torch.float32")

    def assert_rejected_before_sharding(self, model, **changes):
        self.shards.clear()
        args = dict(mesh_policy="world16-fsdp", separate_gan_heads=True)
        args.update(changes)
        with self.assertRaises((ValueError, AssertionError)):
            policy.apply_blockwise_fsdp2_policy(model, **args)
        self.assertEqual(self.shards, [])

    def test_typed_scope_mesh_world_and_40_block_gate(self):
        for value in (0, 1, None, "on", []):
            with self.subTest(enabled=value):
                self.assert_rejected_before_sharding(Model(), separate_gan_heads=value)
        self.assert_rejected_before_sharding(Model(), mesh_policy="node-hsdp")
        self.assert_rejected_before_sharding(Model(), world_size=8)
        self.assert_rejected_before_sharding(Model(30), expected_block_count=30)
        self.assert_rejected_before_sharding(Model(39))
        for value in (False, 1, "yes"):
            model = Model(); model.is_use_gan = value
            self.assert_rejected_before_sharding(model)

    def test_original_request_validator_rejects_non_gan_batch_and_packing(self):
        helper = importlib.import_module("distillschedule.runtime.fsdp2_gan_head_groups")
        base = dict(enabled=True, variant="gan", mesh_policy="world16-fsdp", rollout_batch_size=1, pack_real_score_pair=False)
        helper.validate_gan_head_fsdp_request(**base)
        for changed in ({"variant": "no-gan"}, {"mesh_policy": "node-hsdp"}, {"rollout_batch_size": 2}, {"pack_real_score_pair": True}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                helper.validate_gan_head_fsdp_request(**(base | changed))

    def test_head_alias_parameter_alias_registration_and_type_fail_closed(self):
        cases = []
        m = Model(); m.gan_final_head = m.gan_heads["5"]; cases.append(m)
        m = Model(); m.gan_final_head.weights = list(m.gan_heads["5"].weights); cases.append(m)
        m = Model(); m.gan_heads["5"].weights.append(("alias", m.gan_heads["5"].weights[0][1])); cases.append(m)
        m = Model(); m.extra_parameters.append(("outside", m.gan_heads["5"].weights[0][1])); cases.append(m)
        m = Model(); m.extra_modules.append(("aliased_head", m.gan_heads["5"])); cases.append(m)
        m = Model(); m.blocks[0] = m.gan_heads["5"]; cases.append(m)
        m = Model(); m.gan_final_head = Block(); cases.append(m)
        m = Model(); m.gan_heads = dict(m.gan_heads); cases.append(m)
        m = Model(); m.gan_heads = ModuleDict(reversed(list(m.gan_heads.items()))); cases.append(m)
        m = Model(); m.gan_final_head.weights = []; cases.append(m)
        for index, model in enumerate(cases):
            with self.subTest(case=index):
                self.assert_rejected_before_sharding(model)


if __name__ == "__main__":
    unittest.main()
