"""Content-bound tiny assets: CPU identity/state tests, never model evidence."""
import json
from pathlib import Path

import pytest
import torch

from distillschedule.workloads.best_known.registry import file_sha256, fingerprint


@pytest.fixture(scope="module")
def frozen_assets(tmp_path_factory):
    """Real tiny pre-cast/negative artifacts, not pretrained model evidence."""
    from safetensors.torch import save_file
    from runtime.precast_checkpoint import convert_pretrained_safetensors
    from runtime.gst_pool_contract import write_negative_prompt_asset
    root = tmp_path_factory.mktemp("helios-admission-assets")
    assets = {"assets": {}}
    for role, subfolder in (("generator", "transformer_ode"), ("score", "transformer")):
        model = root / role
        source = model / subfolder; source.mkdir(parents=True)
        (source / "config.json").write_text('{"scope":"CPU_ASSET_IDENTITY_ONLY"}')
        save_file({"weight": torch.ones(2, 2)}, str(source / "model.safetensors"))
        (source / "diffusion_pytorch_model.safetensors.index.json").write_text(json.dumps({
            "metadata": {"total_size": 16}, "weight_map": {"weight": "model.safetensors"}}))
        precast = root / (role + "-precast")
        convert_pretrained_safetensors(source, precast, target_dtype=torch.bfloat16)
        assets["assets"][role + "_model"] = {"path": str(model)}
        assets["assets"][role + "_precast"] = {"path": str(precast)}
    negative = root / "negative"
    write_negative_prompt_asset(negative, embedding=torch.zeros(1, 512, 4096, dtype=torch.bfloat16), model_root=root)
    assets["assets"]["negative_prompt"] = {"path": str(negative)}
    return assets

@pytest.fixture(scope="module")
def original_logical(tmp_path_factory):
    from runtime.sparse_ema import SparseShardEMA
    root = tmp_path_factory.mktemp("helios-independent-step10")
    manifest = {"schema_version": "distillschedule.logical_checkpoint.v3", "global_step": 10,
                "source": "INDEPENDENT_CPU_TINY_STATE_NOT_A_PRETRAINED_CHECKPOINT", "components": {},
                "ema": {"optimization_step": 10}}
    for role, step in (("generator", 2), ("critic", 10)):
        model = torch.nn.Linear(1, 1, bias=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-6 if role == "generator" else 4e-7,
                                      betas=(0., .999), eps=1e-8, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
        scheduler.last_epoch = step
        payloads = {}
        for name, value in (("master", .2), ("first_moment", .03), ("second_moment", .004)):
            path = root / f"{role}-{name}.f32"
            torch.tensor([value], dtype=torch.float32).numpy().tofile(path)
            payloads[name] = {"path": path.name, "bytes": path.stat().st_size,
                              "dtype": "torch.float32", "sha256": file_sha256(path)}
        manifest["components"][role] = {"parameter_version": step, "optimizer_step": step,
            "scheduler_state": scheduler.state_dict(), "logical_numel": 1, "parameter_names": ["weight"],
            "records": {"weight": {"shape": [1], "numel": 1, "offset_elements": 0}}, "payloads": payloads}
    module = torch.nn.Module()
    module.register_parameter("weight", torch.nn.Parameter(torch.tensor([.2])))
    ema = SparseShardEMA(module.named_parameters(), decay=.99, update_after_step=750,
                         base_model="CPU_TINY_FIXTURE_NOT_PRETRAINED")
    ema.optimization_step = 10
    ema.save_pretrained(root / "ema")
    manifest["manifest_hash"] = fingerprint(manifest)
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


def portable_asset_document(frozen, logical):
    rows = {}
    for role, row in frozen["assets"].items():
        identity = {"generator_model": "transformer_ode/config.json", "score_model": "transformer/config.json",
                    "generator_precast": "distillschedule_precast_manifest.json",
                    "score_precast": "distillschedule_precast_manifest.json",
                    "negative_prompt": "manifest.json"}[role]
        rows[role] = {"path": row["path"], "identity_file": identity,
                     "sha256": file_sha256(Path(row["path"]) / identity)}
    rows["logical_checkpoint"] = {"path": str(logical), "identity_file": "manifest.json",
        "sha256": file_sha256(logical / "manifest.json"), "ema_manifest": {
            "identity_file": "ema/sparse_ema_manifest.json",
            "sha256": file_sha256(logical / "ema/sparse_ema_manifest.json")}}
    return {"schema_version": "distillschedule.helios_reference_assets.v1", "assets": rows}
