"""CPU admission of explicit fresh native previous/current AV cached records."""
import json
from pathlib import Path

from distillschedule.core.contracts import require
from distillschedule.strategies.h3_ar_assets import H3ARAssets, HELPER_PINS, checked, validate_source_tree


def prepare_inputs(sources, output, *, assets_path, cycles, seed):
    # Preparation needs the pinned original parser, not a historical cache or
    # a loaded model. The completed manifest becomes assets.cache_manifest for
    # training; requiring it to exist here would be a circular dependency.
    assets = H3ARAssets.from_dict(json.loads(Path(assets_path).read_text()))
    validate_source_tree(Path(assets.source_root).resolve(), assets.source_receipt, assets.source_files)
    require({name: ref.get("sha256") for name, ref in assets.helper_files.items()} == HELPER_PINS
            and all(checked(ref).stem == name for name, ref in assets.helper_files.items()),
            "only accepted native helper paths may enter parser imports", path="h3_ar.data")
    assets.install_import_paths()
    import torch
    require(not torch.cuda.is_initialized(), "fresh W6 preparation must be CPU only", path="h3_ar.data")
    native = assets.source_module("h3_dmd2")
    from distillschedule.training.h3_ar_inputs import prepare_stream
    return prepare_stream(sources, output, native=native, torch=torch, cycles=cycles, seed=seed,
                          pre_publication_guard=assets.audit_imports)
