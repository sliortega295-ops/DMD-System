"""Opt-in original Wan2.1 codec, separately pinned from frozen LiveWan v1.

Only the original vae.py and source license execute through PinnedSource. Safe
local FP32 weights, CPU placement and admission wrap the unmodified WanVAE_
constructor/decode body; unsafe torch.load/CUDA-autocast wrappers are not used.
No weights are shipped/fetched and random fixtures are never called pretrained.
"""
from __future__ import annotations

import ast
from copy import deepcopy
import importlib
import importlib.metadata
from pathlib import Path

from distillschedule.workloads.best_known.registry import file_sha256, fingerprint
from .livewan_public import require, read_json, checked_pin, _safe_shapes, topology

SCHEMA = "distillschedule.original_wan21_cpu_fp32_decoder.v1"
REPOSITORY = "https://github.com/Wan-Video/Wan2.1.git"
REVISION = "9737cba9c1c3c4d04b33fcad41c111989865d315"
SOURCE_FILES = {"wan/modules/vae.py": "4912ffd7a9fbb17ceef9a17d4a9c4b8aee36ec37c32594f54e57693b29909498",
                "LICENSE.txt": "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"}
FIXTURE = "SYNTHETIC_RANDOM_ORIGINAL_WAN21_CODEC_NOT_PRETRAINED"
LOCAL = "LOCAL_ORIGINAL_WAN21_CODEC_WEIGHTS_UNVERIFIED_QUALITY"
SCOPE = "CPU1_FP32_ORIGINAL_WAN21_DECODE_32PX_F81_NOT_QUALITY"
PRECISION = "CPU_FP32_NO_AUTOCAST"
ARCHITECTURE = dict(dim=96, z_dim=16, dim_mult=[1, 2, 4, 4], num_res_blocks=2,
    attn_scales=[], temperal_downsample=[False, True, True], dropout=0.0)
NORMALIZATION = dict(policy="ORIGINAL_WAN21_MEAN_STD_APPLIED_ONCE_INSIDE_NATIVE_DECODE",
    mean=[-.7571, -.7089, -.9113, .1075, -.1745, .9653, -.1517, 1.5508, .4134, -.0715, .5517, -.3632, -.1922, -.9497, .2503, -.2921],
    std=[2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, 3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160])
POLICY = dict(version="livewan_demo_decode_only_clamp_v1", latent_copy_clamp=[-4.0, 4.0],
    native_output_clamp=[-1.0, 1.0], pixel_conversion="((B3THW[0].THWC+1)/2*255).clamp(0,255).uint8",
    fps=16, raw_latents="UNCHANGED")
GEOMETRY = dict(latent_layout="BCTHW", latent_shape=[1, 16, 21, 4, 4],
                video_layout="THWC", video_shape=[81, 32, 32, 3], spatial_stride=8, temporal_stride=4)


def runtime_identity():
    from .livewan_inference import content_runtime
    result = content_runtime()
    module = importlib.import_module("einops")
    result["einops"] = dict(version=importlib.metadata.version("einops"), imported_version=str(module.__version__),
                            entry_sha256=file_sha256(Path(module.__file__).resolve()))
    return result


def source_spec(root):
    return dict(repository=REPOSITORY, revision=REVISION, source_root=str(Path(root).absolute()), files=deepcopy(SOURCE_FILES))


def _source(spec):
    from .upstream import PinnedSource
    require(isinstance(spec, dict) and set(spec) == {"repository", "revision", "source_root", "files"}
            and spec["repository"] == REPOSITORY and spec["revision"] == REVISION
            and spec["files"] == SOURCE_FILES, "separate original Wan2.1 codec source/revision/license closure required")
    root = Path(spec["source_root"])
    require(isinstance(spec["source_root"], str) and root.is_absolute() and root.resolve() == root and root.is_dir(),
            "canonical explicit local codec source root required")
    require(all((root / name).resolve() == root / name for name in SOURCE_FILES), "codec source symlink refused")
    loader = PinnedSource(root, SOURCE_FILES)
    # Check our explicit scale/config against the actual original constructor,
    # without invoking its unrestricted pickle reader or CUDA autocast wrapper.
    tree = ast.parse((root / "wan/modules/vae.py").read_bytes())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "WanVAE")
    constructor = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    values = {node.targets[0].id: ast.literal_eval(node.value) for node in constructor.body
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in {"mean", "std"}}
    require(values == {key: NORMALIZATION[key] for key in ("mean", "std")}, "native codec normalization constructor changed")
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_video_vae")
    call = next(node.value for node in fn.body if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "cfg")
    original = {item.arg: 16 if isinstance(item.value, ast.Name) and item.value.id == "z_dim"
                else ast.literal_eval(item.value) for item in call.keywords}
    require(original == ARCHITECTURE, "native codec architecture constructor changed")
    return loader, loader.import_module("wan.modules.vae")


def admit_manifest(pin):
    """Readmit closure/runtime/config/header/full finite safe state before use."""
    import torch
    from safetensors.torch import load_file
    from .checkpoint import rng_state, restore_rng
    topology()
    manifest = read_json(checked_pin(pin))
    checked_pin(pin)
    require(isinstance(manifest, dict) and set(manifest) == {"schema_version", "scope", "precision", "source", "runtime",
        "architecture", "normalization", "decode_policy", "geometry", "evidence_kind", "weight_provenance", "weights", "budget"}
        and manifest["schema_version"] == SCHEMA and manifest["scope"] == SCOPE
        and manifest["precision"] == PRECISION and manifest["normalization"] == NORMALIZATION
        and manifest["decode_policy"] == POLICY and manifest["geometry"] == GEOMETRY,
        "explicit original Wan2.1 codec schema/normalization/layout/decode policy required")
    require(manifest["runtime"] == runtime_identity(), "codec runtime changed")
    kind, architecture, provenance = manifest["evidence_kind"], manifest["architecture"], manifest["weight_provenance"]
    require(kind in (FIXTURE, LOCAL) and isinstance(provenance, dict) and set(provenance) == {"status", "origin", "license"}
            and all(isinstance(v, str) and bool(v.strip()) for v in provenance.values())
            and provenance["status"] == ("RANDOM_NOT_PRETRAINED" if kind == FIXTURE else "USER_SUPPLIED_NOT_QUALITY_VERIFIED"),
            "truthful explicit codec weight provenance/license required")
    expected = deepcopy(ARCHITECTURE)
    if kind == FIXTURE:
        require(isinstance(architecture, dict) and type(architecture.get("dim")) is int
                and 2 <= architecture["dim"] <= 16 and architecture["dim"] % 2 == 0,
                "bounded random native-class fixture width required")
        expected["dim"] = architecture["dim"]
    require(architecture == expected, "original codec config changed; no production scaled-width decoder")
    limits = dict(max_parameters=200_000_000, max_loaded_bytes=2_000_000_000,
                  max_activation_elements=100_000_000, max_attention_elements=8_000_000)
    budget = manifest["budget"]
    require(isinstance(budget, dict) and set(budget) == set(limits)
            and all(type(budget[k]) is int and 0 < budget[k] <= v for k, v in limits.items()),
            "explicit bounded codec parameter/loaded/activation/attention capacity required")
    require(16_384*architecture["dim"] <= budget["max_activation_elements"]
            and 2*4*(4*4)**2 <= budget["max_attention_elements"], "codec activation/attention capacity exceeded")
    loader, api = _source(manifest["source"])
    before = rng_state()
    try:
        with torch.device("meta"):
            probe = api.WanVAE_(**architecture)
        expected_shapes = {key: tuple(t.shape) for key, t in probe.state_dict().items()}
        count = sum(t.numel() for t in probe.state_dict().values())
        require(0 < count <= budget["max_parameters"]
                and 4*(2*count+budget["max_activation_elements"]) <= budget["max_loaded_bytes"],
                "codec full architecture/strict-load capacity exceeded")
    finally:
        restore_rng(before)
    require(type(manifest["weights"]) is list and 0 < len(manifest["weights"]) <= 16, "explicit safe codec weight shards required")
    shapes, state, paths, loaded = {}, {}, set(), 0
    for weight in manifest["weights"]:
        path = checked_pin(weight)
        require(str(path) not in paths, "duplicate codec weight shard")
        paths.add(str(path))
        current = _safe_shapes(weight, evidence_kind=kind)
        require(not shapes.keys() & current.keys(), "duplicate codec state keys")
        shapes.update(current)
        loaded += weight["bytes"]
        require(loaded <= budget["max_loaded_bytes"], "codec weight byte capacity exceeded")
    require(shapes == expected_shapes, "complete original codec state layout changed")
    for weight in manifest["weights"]:
        raw = load_file(str(checked_pin(weight)), device="cpu")
        checked_pin(weight)
        require(all(t.dtype == torch.float32 and t.device.type == "cpu" and bool(t.isfinite().all()) for t in raw.values()),
                "finite all-FP32 supplied codec state required; no casting")
        state.update(raw)
    loader.validate()
    checked_pin(pin)
    return manifest, loader, api, state


def decode_video(generated, pin):
    import torch
    from .checkpoint import rng_state, restore_rng
    from .inference import latent_outputs
    from .livewan_inference import latent_schema
    require(latent_outputs(generated) == latent_schema(), "codec requires exact native BCTHW FP32 latent21")
    latents = generated["latents"]
    require(latents.device.type == "cpu" and latents.dtype == torch.float32, "codec exact CPU FP32 input required")
    manifest, loader, api, state = admit_manifest(pin)
    before = rng_state()
    try:
        # Construct only the actual original class. Its unsafe load/autocast
        # conveniences are never invoked, and no math methods are replaced.
        with torch.device("cpu"):
            model = api.WanVAE_(**manifest["architecture"])
        model.requires_grad_(False).eval()
        model.load_state_dict(state, strict=True)
        mean = torch.tensor(NORMALIZATION["mean"], dtype=torch.float32)
        std = torch.tensor(NORMALIZATION["std"], dtype=torch.float32)
        with torch.no_grad():
            decoded = model.decode(latents.clamp(-4.0, 4.0), [mean, 1.0/std])
        require(decoded.shape == (1, 3, 81, 32, 32) and decoded.dtype == torch.float32
                and decoded.device.type == "cpu" and bool(decoded.isfinite().all()), "original codec output shape/dtype/nonfinite")
        require(all(value is None for value in model._feat_map), "native codec feature-cache clear failed")
        frames = ((decoded.clamp(-1.0, 1.0)[0].permute(1, 2, 3, 0)+1.0)/2.0*255.0).clamp(0, 255).to(torch.uint8)
    finally:
        restore_rng(before)
    loader.validate()
    checked_pin(pin)
    for weight in manifest["weights"]:
        checked_pin(weight)
    identity = dict(schema_version=SCHEMA, manifest=deepcopy(pin), source=deepcopy(manifest["source"]),
        importer=deepcopy(loader.identity), runtime=deepcopy(manifest["runtime"]), architecture=deepcopy(manifest["architecture"]),
        normalization=deepcopy(NORMALIZATION), decode_policy=deepcopy(POLICY),
        weights=deepcopy(manifest["weights"]), weight_provenance=deepcopy(manifest["weight_provenance"]),
        evidence_kind=manifest["evidence_kind"], bridge_sha256=file_sha256(Path(__file__)),
        identity_sha256=fingerprint(manifest), quality_validation="NOT_RUN", claimable_speedup=False)
    return dict(video=frames.contiguous(), decoder=identity)
