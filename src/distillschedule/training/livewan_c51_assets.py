"""Explicit current-file assets for the full C51 continuation, not old runs.

The original reference asset validator is reused for licensed model/source
inventories and FP32/BF16 precision. Its FA2-only execution API is never called.
The additional two kernel roots must contain the unchanged C51 artifacts.
"""
from copy import deepcopy
import hashlib
import importlib.metadata
import os
from pathlib import Path
import subprocess
import sys

from distillschedule.workloads.best_known.registry import file_sha256, fingerprint
from .livewan_reference_assets import (
    BACKEND, DEPENDENCIES, ReferenceAssets, asset_config as native_assets, _asset_config,
    environment_snapshot, require,
)

SCHEMA = "distillschedule.livewan_c51_assets.v1"
HF_SCHEMA = "distillschedule.livewan_c51_assets.v2"
KERNEL_ENV = {"fa2": "DISTILLSCHEDULE_FLASH_ATTN_KERNEL", "fa3": "DISTILLSCHEDULE_SCORE_FA3_KERNEL"}


def uses_hf_provider(value):
    return type(value) is dict and value.get("schema_version") == HF_SCHEMA


def kernel_files(*, hf_provider=False):
    if hf_provider:
        from distillschedule.runtime.livewan_c51_provider import provider_files
        return provider_files()
    from distillschedule.runtime import livewan_attention_backend as fa2, livewan_score_fa3 as fa3
    return {"fa2": {"metadata.json": fa2.EXPECTED_METADATA_SHA256,
                    "flash_attn2/__init__.py": fa2.EXPECTED_PACKAGE_SHA256,
                    "_flash_attn2_cuda_a511a4c.abi3.so": fa2.EXPECTED_EXTENSION_SHA256},
            "fa3": dict(fa3.HASHES)}


def overlay_inventory(native, root):
    """Verify the SAME original LiveWan Wan2.1 attention/config overlay.

    The existing public prepare_livewan_crossscale_source.sh can produce it.
    This helper neither copies sources nor invents a clean Git identity for
    the patched tree: every unmodified file binds the real clean Wan gitlink;
    the two patches bind actual files at the real clean LiveWan gitlink.
    """
    root = Path(root).expanduser().resolve()
    expected = dict(native["sources"]["wan21"]["files"])
    livewan = native["sources"]["livewan"]
    for destination, original in (("wan/modules/attention.py", "wan21_patches/modules/attention.py"),
                                   ("wan/configs/__init__.py", "wan21_patches/configs/__init__.py")):
        pinned = subprocess.check_output(["git", "-C", livewan["root"], "show", livewan["head"]+":"+original],
                                         stderr=subprocess.PIPE)
        sha = hashlib.sha256(pinned).hexdigest()
        require(file_sha256(Path(livewan["root"])/original) == sha, "original C51 Wan overlay patch changed")
        expected[destination] = sha
    require(root.is_dir() and {str(p.relative_to(root)) for p in (root/"wan").rglob("*.py")} ==
            {p for p in expected if p.endswith(".py")}, "C51 Wan overlay Python inventory changed")
    for name, sha in expected.items():
        path = root/name
        require(path.is_file() and path.resolve().is_relative_to(root) and file_sha256(path) == sha,
                "C51 original Wan overlay source missing/changed: " + name)
    return dict(root=str(root), files=expected)


def source_origins(value):
    """Reject foreign/preloaded imports, including clean-but-unpatched Wan."""
    result = {}
    for name, module in tuple(sys.modules.items()):
        origin = (value["native"]["sources"]["livewan"] if name == "wanstreamer" or name.startswith("wanstreamer.")
                  else value["wan21_overlay"] if name == "wan" or name.startswith("wan.") else None)
        if origin is None:
            continue
        path = Path(getattr(module, "__file__", "")).resolve()
        require(path.is_relative_to(origin["root"]) and origin["files"].get(str(path.relative_to(origin["root"]))) == file_sha256(path),
                "C51 actual original overlay import origin/hash changed: " + name)
        result[name] = dict(path=str(path), sha256=file_sha256(path))
    if uses_hf_provider(value):
        from distillschedule.runtime.livewan_c51_provider import source_origins as provider_origins
        result["attention_provider"] = provider_origins(value["kernels"])
    return result


def provider_binding(value):
    """Stable, complete expected provider identity consumed by owners/plans."""
    require(uses_hf_provider(value), "C51 HF provider requires explicit v2 assets")
    return dict(provider=deepcopy(value["attention_provider"]), kernels=deepcopy(value["kernels"]))


def asset_config(value, *, verify_payloads=False):
    hf = uses_hf_provider(value)
    fields = {"schema_version", "native", "kernels", "wan21_overlay"}
    require(type(value) is dict and set(value) == fields | ({"attention_provider"} if hf else set())
            and value["schema_version"] in (SCHEMA, HF_SCHEMA), "explicit full C51 current assets required")
    if hf:
        from distillschedule.runtime.livewan_c51_provider import provider_config
        provider_config(value["attention_provider"])
        _asset_config(value["native"], verify_payloads=verify_payloads, dependencies=DEPENDENCIES - {"flash-attn"})
    else:
        native_assets(value["native"], verify_payloads=verify_payloads)
    require(type(value["wan21_overlay"]) is dict and set(value["wan21_overlay"]) == {"root", "files"}
            and Path(value["wan21_overlay"]["root"]).is_absolute()
            and value["wan21_overlay"] == overlay_inventory(value["native"], value["wan21_overlay"]["root"]),
            "C51 explicit original Wan attention/config overlay differs")
    require(type(value["kernels"]) is dict and set(value["kernels"]) == {"fa2", "fa3"},
            "both explicit C51 FA2 and FA3 roots required")
    for kind, hashes in kernel_files(hf_provider=hf).items():
        row = value["kernels"][kind]
        require(type(row) is dict and set(row) == {"root", "files"}
                and row["files"] == hashes and Path(row["root"]).is_absolute()
                and str(Path(row["root"]).resolve()) == row["root"], "C51 kernel identity/root changed")
        for name, sha in hashes.items():
            path = Path(row["root"]) / name
            # Extension files are checked too, even at CPU admission. No import
            # or inferred historical mount is needed to inspect these bytes.
            require(path.is_file() and file_sha256(path) == sha, "C51 pinned kernel file missing/changed: " + str(path))
        if hf:
            actual = {str(p.relative_to(row["root"])) for p in Path(row["root"]).rglob("*")
                      if p.is_file() and (p.suffix == ".py" or p.name.endswith(".so"))}
            require(actual == {name for name in hashes if name.endswith((".py", ".so"))},
                    "C51 HF complete Python/extension inventory changed")
    return deepcopy(value)


def admit_runtime(value):
    import torch
    hf = uses_hf_provider(value)
    if hf:
        asset_config(value)  # no unchecked runtime-only bypass of full provider/source pins
    runtime = value["native"]["runtime"]
    actual_env = environment_snapshot(os.environ)
    for kind, name in KERNEL_ENV.items():
        require(actual_env.pop(name, None) == value["kernels"][kind]["root"],
                "explicit effective C51 kernel environment/root differs: " + name)
    changed = sorted(name for name in actual_env.keys() | runtime["environment"].keys()
                     if actual_env.get(name) != runtime["environment"].get(name))
    require(actual_env == runtime["environment"],
            "effective C51 precision/environment controls changed: " + ", ".join(changed))
    require(str(torch.__version__) == "2.10.0+cu128", "C51 requires validated Torch 2.10 CUDA 12.8")
    for name, expected in runtime["dependency_versions"].items():
        require(importlib.metadata.version(name) == expected, "C51 dependency changed: " + name)
    actual = {**BACKEND, "default_dtype": str(torch.get_default_dtype()),
              "matmul_tf32": torch.backends.cuda.matmul.allow_tf32, "cudnn_tf32": torch.backends.cudnn.allow_tf32,
              "matmul_precision": torch.get_float32_matmul_precision(),
              "deterministic": torch.are_deterministic_algorithms_enabled(), "cudnn_benchmark": torch.backends.cudnn.benchmark}
    require(actual == runtime["backend"], "C51 original precision/reduction/backend controls changed")
    result = {"dependencies": deepcopy(runtime["dependency_versions"]), "precision": actual,
              "attention": "original_G_FA2_splitKV__Score_FA3_exact_SAC", "kernels": deepcopy(value["kernels"])}
    if hf:
        from distillschedule.runtime.livewan_c51_provider import runtime_identity
        result["attention_provider"] = runtime_identity(value["attention_provider"], value["kernels"], torch=torch)
    return result


class C51Assets(ReferenceAssets):
    """Same native bootstrap ABI, separately admitted complete C51 identity."""
    @classmethod
    def load(cls, value):
        admitted = asset_config(value, verify_payloads=True)
        result = cls(admitted["native"])
        # ReferenceAssets is frozen; the native model builder only reads value.
        object.__setattr__(result, "c51_document", admitted)
        return result

    @property
    def sha256(self):
        return fingerprint(self.c51_document) if uses_hf_provider(self.c51_document) else super().sha256

    @property
    def value(self):
        result = super().value
        if uses_hf_provider(self.c51_document):
            result["attention_provider"] = provider_binding(self.c51_document)
        return result

    def native_plan(self):
        result = super().native_plan()
        if uses_hf_provider(self.c51_document):
            result["evidence"] = dict(attention_provider=provider_binding(self.c51_document))
        return result

    def unchanged(self):
        require(asset_config(self.c51_document, verify_payloads=True) == self.c51_document,
                "full C51 asset/source payload drift")
        if uses_hf_provider(self.c51_document):
            from distillschedule.runtime.livewan_c51_provider import source_origins as provider_origins
            provider_origins(self.c51_document["kernels"])
