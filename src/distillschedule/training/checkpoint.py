"""Complete-cycle checkpoints for framework-owned training state.

Native distributed adapters keep their own shard writers. Program sessions use
this same-topology atomic writer, with one safe local shard per actual rank.
"""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time
import uuid
from distillschedule.workloads.best_known.registry import file_sha256 as _sha
from . import distributed
from .tensor_state import pack, decode, layout, live_meshes

SCHEMA = "distillschedule.training_checkpoint.v2"
_STORAGE_VISIBILITY_TIMEOUT = 30.0


def _await_shared_paths(paths, label, *, directory=False, hashes=None):
    """Wait only for visibility whose producer already completed collectively.

    A process-group barrier is not a shared-filesystem metadata barrier. Never
    create a missing peer directory or accept an incomplete checkpoint here.
    During save publication, content must match the exact writer-attested hash,
    even when stale/partial reads are nonzero. Ordinary restart never retries.
    Hashes accept exact Path keys, or legacy basenames for flat inventories.
    """
    timeout = 0.0
    if distributed.rank_world()[1] > 1:
        from distillschedule.open_workload.replicated import metadata_timeout_seconds
        # Peers can already be waiting in checked()/primary() while we poll.
        # Leave most of the actual metadata timeout for I/O and that collective.
        timeout = min(_STORAGE_VISIBILITY_TIMEOUT, metadata_timeout_seconds() / 4)
    deadline = time.monotonic() + timeout
    pending = list(paths)
    while True:
        missing = [path for path in pending if not (path.is_dir() if directory else path.is_file())]
        mismatches = []
        if hashes is not None:
            for path in pending:
                key = path if path in hashes else path.name
                if path in missing or key not in hashes:
                    continue
                try:
                    # Stream model-sized peer shards; never materialize them.
                    observed = _sha(path)
                except FileNotFoundError:
                    missing.append(path)
                    continue
                expected = hashes[key]
                if observed == expected:
                    continue
                mismatches.append(f"{path.name}: hash changed (expected {expected}, observed {observed})")
                missing.append(path)
        if not missing:
            return
        if time.monotonic() >= deadline:
            raise ValueError(f"{label}: shared storage visibility timed out: {missing}; {mismatches}")
        # Writers have finished and these paths are immutable. Do not rehash
        # already verified shards while waiting for another peer's visibility.
        pending = missing
        time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))


def cpu_tree(value):
    import torch
    if isinstance(value, torch.Tensor):
        from distillschedule.open_workload.torch_backend import full_tensor
        value = full_tensor(value)
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if value is None or type(value) in (str, bool, int, float):
        return value
    raise ValueError(f"checkpoint state needs an explicit serializer: {type(value).__name__}")


def _class(value):
    return type(value).__module__ + "." + type(value).__qualname__


def _parameter_names(state):
    import torch
    from distillschedule.open_workload.torch_backend import unwrap_module
    return {id(p): f"{key}.{name}" for key, value in state.items()
            if isinstance(value, torch.nn.Module)
            for name, p in unwrap_module(value).named_parameters()}


def pack_state(state):
    import torch
    from distillschedule.open_workload.torch_backend import unwrap_module
    parameter_names = _parameter_names(state)
    packed = {}
    for key, value in state.items():
        if isinstance(value, torch.nn.Module):
            value = unwrap_module(value)
            packed[key] = {"kind": "module", "class": _class(value),
                           "value": pack(value.state_dict()),
                           "training": {name: child.training for name, child in value.named_modules()},
                           "requires_grad": {name: p.requires_grad for name, p in value.named_parameters(remove_duplicate=False)}}
        elif isinstance(value, torch.optim.Optimizer):
            packed[key] = {"kind": "optimizer", "class": _class(value), "value": pack(value.state_dict()),
                           "parameters": [[parameter_names[id(p)] for p in group["params"]]
                                          for group in value.param_groups]}
        elif callable(getattr(value, "state_dict", None)) and callable(getattr(value, "load_state_dict", None)):
            packed[key] = {"kind": "stateful", "class": _class(value), "value": pack(value.state_dict())}
        else:
            packed[key] = {"kind": "value", "value": pack(value)}
    return packed


def _validate_layout(current, restored, name):
    """Validate module extra-state with the same explicit safe-tree contract."""
    import torch
    from torch.distributed.tensor import DTensor
    if isinstance(current, torch.Tensor):
        if not isinstance(restored, torch.Tensor) or current.shape != restored.shape or current.dtype != restored.dtype:
            raise ValueError(f"checkpoint module shape/dtype changed: {name}")
        if isinstance(current, DTensor) != isinstance(restored, DTensor) or (
                isinstance(current, DTensor) and layout(current) != layout(restored)):
            raise ValueError(f"checkpoint tensor layout changed: {name}")
    elif isinstance(current, dict):
        if not isinstance(restored, dict) or current.keys() != restored.keys():
            raise ValueError(f"checkpoint module keys changed: {name}")
        for key, value in current.items():
            _validate_layout(value, restored[key], f"{name}.{key}")
    elif isinstance(current, (tuple, list)):
        if type(restored) is not type(current) or len(current) != len(restored):
            raise ValueError(f"checkpoint module sequence layout changed: {name}")
        for index, (a, b) in enumerate(zip(current, restored)):
            _validate_layout(a, b, f"{name}.{index}")
    elif type(current) not in (type(None), str, bool, int, float) or type(restored) is not type(current):
        raise ValueError(f"checkpoint module extra-state type changed: {name}")


def _validate_optimizer_layout(optimizer, restored, name):
    """Check known pointwise slots, not native factored/flattened state.

    The native loader owns other optimizer schemas. Exact types avoid imposing
    a built-in schema on custom subclasses. Validate the decoded source slots:
    load_state_dict may otherwise silently cast a malformed moment's dtype.
    """
    import torch
    fields = {
        torch.optim.Adam: ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"),
        torch.optim.AdamW: ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"),
        torch.optim.SparseAdam: ("exp_avg", "exp_avg_sq"),
        torch.optim.SGD: ("momentum_buffer",),
        torch.optim.Adadelta: ("square_avg", "acc_delta"),
        torch.optim.Adagrad: ("sum",),
        torch.optim.Adamax: ("exp_avg", "exp_inf"),
        torch.optim.ASGD: ("ax",),
        torch.optim.NAdam: ("exp_avg", "exp_avg_sq"),
        torch.optim.RAdam: ("exp_avg", "exp_avg_sq"),
        torch.optim.RMSprop: ("square_avg", "momentum_buffer", "grad_avg"),
        torch.optim.Rprop: ("prev", "step_size"),
        torch.optim.Adafactor: ("variance",),
    }.get(type(optimizer), ())
    for group, saved_group in zip(optimizer.param_groups, restored["param_groups"]):
        for parameter, parameter_id in zip(group["params"], saved_group["params"]):
            values = restored["state"].get(parameter_id, {})
            for field in fields:
                if field in values:
                    _validate_layout(parameter, values[field], f"{name}.{field}")


def _stage_state(packed, state, *, progress=None):
    import torch
    from copy import copy
    from distillschedule.open_workload.torch_backend import unwrap_module
    if set(packed) != set(state):
        raise ValueError("checkpoint state names changed")
    meshes = live_meshes(state)
    restored = {key: decode(row["value"], meshes) for key, row in packed.items()}
    parameter_names = _parameter_names(state)
    optimizers = {}
    # Validate/load staged optimizer objects before mutating any live tensor.
    for key, row in packed.items():
        value = unwrap_module(state[key]) if isinstance(state[key], torch.nn.Module) else state[key]
        expected_kind = ("module" if isinstance(value, torch.nn.Module) else "optimizer"
                         if isinstance(value, torch.optim.Optimizer) else "stateful"
                         if callable(getattr(value, "state_dict", None)) and callable(getattr(value, "load_state_dict", None))
                         else "value")
        if row.get("kind") != expected_kind:
            raise ValueError(f"checkpoint state kind changed: {key}")
        if row["kind"] != "value" and row["class"] != _class(value):
            raise ValueError(f"checkpoint state class changed: {key}")
        if row["kind"] == "module":
            current = value.state_dict()
            _validate_layout(current, restored[key], key)
            if row["requires_grad"] != {name: p.requires_grad for name, p in value.named_parameters(remove_duplicate=False)}:
                raise ValueError(f"checkpoint trainables changed: {key}")
            if (set(row["training"]) != {name for name, _ in value.named_modules()}
                    or any(type(item) is not bool for item in row["training"].values())):
                raise ValueError(f"checkpoint module training modes changed: {key}")
        elif row["kind"] == "optimizer":
            expected = [[parameter_names[id(p)] for p in group["params"]] for group in value.param_groups]
            if row.get("parameters") != expected:
                raise ValueError(f"checkpoint optimizer parameter ownership changed: {key}")
            staged = copy(value)
            staged.param_groups = [dict(group, params=list(group["params"])) for group in value.param_groups]
            staged.state = {}
            staged.load_state_dict(restored[key])
            _validate_optimizer_layout(staged, restored[key], key)
            optimizers[key] = staged
        elif row["kind"] == "value" and isinstance(value, torch.Tensor):
            _validate_layout(value, restored[key], key)
        elif row["kind"] == "stateful" and callable(getattr(value, "validate_state_dict", None)):
            value.validate_state_dict(restored[key])
        if row["kind"] == "stateful" and callable(getattr(value, "validate_checkpoint_progress", None)):
            value.validate_checkpoint_progress(restored[key], progress)
    return restored, optimizers


def unpack_state(packed, state, *, progress=None):
    import torch
    from distillschedule.open_workload.torch_backend import unwrap_module
    # No rank begins mutation while any rank rejects a shard/layout/optimizer.
    restored, optimizers = distributed.checked(lambda: _stage_state(packed, state, progress=progress), "checkpoint state admission")
    for key, row in packed.items():
        value = unwrap_module(state[key]) if isinstance(state[key], torch.nn.Module) else state[key]
        if row["kind"] == "module":
            value.load_state_dict(restored[key], strict=True)
            for name, child in value.named_modules():
                child.training = row["training"][name]
        elif row["kind"] == "optimizer":
            value.state = optimizers[key].state
            value.param_groups = optimizers[key].param_groups
        elif row["kind"] == "stateful":
            value.load_state_dict(restored[key])
        elif isinstance(value, torch.Tensor):
            with torch.no_grad():
                value.copy_(restored[key].to(value.device))
        else:
            state[key] = deepcopy(restored[key])


def rng_state(*, device=None):
    import torch
    result = {"python": random.getstate(), "torch": torch.get_rng_state()}
    owned = torch.device(device) if device is not None else None
    if (owned is not None and owned.type == "cuda") or (owned is None and torch.cuda.is_initialized()):
        # The execution ABI owns one device per rank. Querying all visible
        # devices can initialize unrelated CUDA contexts and consume their HBM.
        index = owned.index if owned is not None and owned.index is not None else torch.cuda.current_device()
        result["cuda"] = {"device": index, "state": torch.cuda.get_rng_state(index)}
    try:
        import numpy as np
        name, values, position, gauss, cached = np.random.get_state()
        result["numpy"] = (name, values.tolist(), position, gauss, cached)
    except ImportError:
        pass
    return result


def validate_rng(value):
    import torch
    random.Random().setstate(value["python"])
    torch.Generator(device="cpu").set_state(value["torch"])
    if "cuda" in value:
        if (not torch.cuda.is_initialized() or value["cuda"]["device"] != torch.cuda.current_device()
                or not isinstance(value["cuda"]["state"], torch.Tensor)
                or value["cuda"]["state"].dtype != torch.uint8
                or value["cuda"]["state"].shape != torch.cuda.get_rng_state(torch.cuda.current_device()).shape):
            raise ValueError("checkpoint CUDA topology changed")
    if "numpy" in value:
        import numpy as np
        name, values, position, gauss, cached = value["numpy"]
        np.random.RandomState().set_state((name, np.asarray(values, dtype=np.uint32), position, gauss, cached))


def restore_rng(value):
    import torch
    validate_rng(value)
    random.setstate(value["python"])
    torch.set_rng_state(value["torch"])
    if "cuda" in value:
        torch.cuda.set_rng_state(value["cuda"]["state"], device=value["cuda"]["device"])
    if "numpy" in value:
        import numpy as np
        name, values, position, gauss, cached = value["numpy"]
        np.random.set_state((name, np.asarray(values, dtype=np.uint32), position, gauss, cached))


def _native_inventory(path, value, *, hashes=False):
    """Validate a native engine's opaque files inside the common publication.

    State/optimizer formats remain owned by that engine, not by this publisher.
    No optional native files can weaken the regular all-rank shard contract.
    """
    if (not isinstance(value, dict) or set(value) != {"schema_version", "owner", "files"}
            or value["schema_version"] != "distillschedule.native_checkpoint_sidecars.v1"
            or not isinstance(value["owner"], dict) or not isinstance(value["files"], dict) or not value["files"]):
        raise ValueError("invalid native checkpoint sidecar inventory")
    root = Path(path).resolve()
    expected = value["files"]
    for name, sha in expected.items():
        candidate = root/name
        if (not isinstance(name, str) or not name.startswith("native/") or ".." in Path(name).parts
                or not candidate.resolve().is_relative_to(root/"native") or candidate.is_symlink()
                or not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha)
                or not candidate.is_file() or hashes and _sha(candidate) != sha):
            raise ValueError("native checkpoint shard missing, unsafe or hash changed: " + str(name))
    observed = {str(p.relative_to(root)) for p in (root/"native").rglob("*") if p.is_file()}
    if observed != set(expected):
        raise ValueError("native checkpoint shard inventory changed")


def save_checkpoint(path, state, *, identity, cycle, cursor, versions, policy=None, data_progress=None,
                    pre_publication_guard=None, native_checkpoint=None):
    import torch
    if type(cycle) is not int or cycle < 0 or type(cursor) is not int or cursor < 0:
        raise ValueError("checkpoint must describe a complete cycle and data cursor")
    def boundary():
        if int(os.environ.get("WORLD_SIZE", "1")) != distributed.rank_world()[1]:
            raise ValueError("checkpoint requires the actual initialized rank group")
        if any(p.grad is not None for value in state.values() if isinstance(value, torch.nn.Module)
               for p in value.parameters()):
            raise ValueError("checkpoint requires a zero-gradient complete-cycle boundary")
        for value in state.values():
            validate = getattr(value, "validate_checkpoint_boundary", None)
            if callable(validate):
                validate()
            progress_hook = getattr(value, "validate_checkpoint_progress", None)
            if callable(progress_hook):
                progress_hook(value.state_dict(), {"cycle":cycle, "cursor":cursor, "versions":dict(versions)})
        if native_checkpoint is not None:
            native_checkpoint.validate_checkpoint_boundary()
    distributed.checked(boundary, "checkpoint boundary")
    rank, world = distributed.rank_world()
    allocated = distributed.topology()
    common = {"identity": identity, "cycle": cycle, "cursor": cursor,
              "versions": dict(versions), "policy": policy, "topology": allocated}
    if data_progress is not None:
        common['data_progress'] = data_progress
    if any(row != common for row in distributed.gather(common)):
        raise ValueError("ranks disagree on checkpoint identity or completed progress")
    path = Path(path).resolve()
    token = distributed.gather(uuid.uuid4().hex if rank == 0 else None)[0]
    staging = path.with_name("." + path.name + ".tmp-" + token)
    def create():
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(path)
        staging.mkdir()
    distributed.primary(create, "checkpoint staging")
    try:
        payload = distributed.checked(lambda: {"state": pack_state(state), "rng": rng_state(), "cycle": cycle,
                   "cursor": cursor, "versions": dict(versions), "policy": policy}, "checkpoint packing")
        if data_progress is not None:
            payload['data_progress'] = data_progress
        filename = "state.pt" if world == 1 else f"rank-{rank:05d}.pt"
        def write_shard():
            _await_shared_paths([staging], "checkpoint output must be shared by all ranks", directory=True)
            with (staging / filename).open("xb") as handle:
                torch.save(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())
            return {filename: _sha(staging / filename)}
        shard = distributed.checked(write_shard, "checkpoint shard writing")
        files = {key: value for row in distributed.gather(shard) for key, value in row.items()}
        native = None
        if native_checkpoint is not None:
            # The actual engine performs its own collective native save. It
            # returns only hashes/metadata, never a parallel optimizer format.
            local = native_checkpoint.save(staging/"native")
            native_rows = distributed.gather(local)
            def native_admission():
                owners = [row["owner"] for row in native_rows]
                if any(owner != owners[0] for owner in owners):
                    raise ValueError("native rank checkpoint owners disagree")
                inventory = {}
                for row in native_rows:
                    for name, sha in row["files"].items():
                        if name in inventory and inventory[name] != sha:
                            raise ValueError("native rank checkpoint hashes disagree")
                        inventory[name] = sha
                value = dict(schema_version="distillschedule.native_checkpoint_sidecars.v1",
                             owner=owners[0], files=inventory)
                _await_shared_paths([staging/name for name in inventory], "native checkpoint shared shard visibility",
                                    hashes={staging/name: sha for name, sha in inventory.items()})
                _native_inventory(staging, value, hashes=True)
                native_checkpoint.validate_inventory(staging/"native", owners[0])
                return value
            native = distributed.checked(native_admission, "native checkpoint inventory admission")
        if pre_publication_guard is not None:
            # Every rank still owns its full CPU snapshot here. Exchange any
            # staging-capacity failure before publishing metadata or _SUCCESS.
            distributed.checked(pre_publication_guard, "checkpoint pre-publication guard")
        def publish():
            if len(files) != world:
                raise ValueError("checkpoint shards incomplete on shared storage")
            # A publisher's local source readback is strict; only completed
            # peer writes may need time to become visible on shared storage.
            if _sha(staging / filename) != files[filename]:
                raise ValueError(f"checkpoint source shard hash changed: {filename}")
            peers = {name: sha for name, sha in files.items() if name != filename}
            _await_shared_paths([staging / name for name in peers],
                                "checkpoint shards incomplete on shared storage", hashes=peers)
            manifest = {"schema_version": SCHEMA, "kind": "RESUMABLE_COMPLETE_CYCLE",
                        "identity": identity, "cycle": cycle, "cursor": cursor,
                        "versions": dict(versions), "policy": policy,
                        "topology": allocated, "torch": torch.__version__, "files": files}
            if data_progress is not None:
                manifest['data_progress'] = data_progress
            if native is not None:
                manifest['native_checkpoint'] = native
            manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
            contents = {"manifest.json": manifest_bytes,
                        "_SUCCESS": (hashlib.sha256(manifest_bytes).hexdigest() + "\n").encode("ascii")}
            metadata = {name: {"size": len(value), "sha256": hashlib.sha256(value).hexdigest()}
                        for name, value in contents.items()}
            # Attest the intended bytes, not a possibly stale filesystem read.
            # Flush and strictly verify the publisher's copy before rename;
            # only consumers may wait for exact publication content visibility.
            for name, value in contents.items():
                with (staging / name).open("xb") as handle:
                    handle.write(value)
                    handle.flush()
                    os.fsync(handle.fileno())
            for name, value in contents.items():
                if (staging / name).read_bytes() != value:
                    raise ValueError(f"checkpoint source metadata hash changed: {name}")
            os.rename(staging, path)
            return metadata
        metadata = distributed.primary(publish, "checkpoint publication")
        metadata = distributed.gather(metadata)[0]
        def visible_publication():
            names = (*files, *(native["files"] if native is not None else ()), "manifest.json", "_SUCCESS")
            _await_shared_paths([path / name for name in names],
                                "checkpoint publication incomplete on shared storage",
                                hashes={name: row["sha256"] for name, row in metadata.items()})
        distributed.checked(visible_publication, "checkpoint shared publication visibility")
    except BaseException:
        if world == 1:
            shutil.rmtree(staging, ignore_errors=True)
        # Distributed incomplete staging remains explicitly unpublished for audit.
        raise
    return path


def read_manifest(path, *, identity):
    """Validate restart metadata before constructing a physical model layout.

    Tensor hashes are checked again before any state restore. Metadata does not
    load a model-sized snapshot simply to read its saved policy and cursor.
    All ranks participate when a distributed group is initialized.
    """
    import torch
    allocated = distributed.topology()
    path = Path(path).resolve()
    if not (path / "_SUCCESS").is_file() or not (path / "manifest.json").is_file():
        raise ValueError("checkpoint is not complete")
    if (path / "_SUCCESS").read_text().strip() != _sha(path / "manifest.json"):
        raise ValueError("checkpoint manifest hash changed")
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("schema_version") != SCHEMA or manifest.get("identity") != identity:
        raise ValueError("checkpoint identity changed")
    if manifest["topology"] != allocated:
        raise ValueError("checkpoint topology changed")
    if manifest.get("torch") != torch.__version__:
        raise ValueError("checkpoint runtime version changed")
    _, world = distributed.rank_world()
    expected = {"state.pt"} if world == 1 else {f"rank-{r:05d}.pt" for r in range(world)}
    if set(manifest.get("files", {})) != expected or any(not (path / name).is_file() for name in expected):
        raise ValueError("checkpoint shard missing or hash changed")
    if "native_checkpoint" in manifest:
        _native_inventory(path, manifest["native_checkpoint"])
    return manifest


def read_checkpoint(path, *, identity):
    import torch
    path = Path(path).resolve()
    manifest = read_manifest(path, identity=identity)
    if any(_sha(path / name) != sha for name, sha in manifest["files"].items()):
        raise ValueError("checkpoint shard missing or hash changed")
    if "native_checkpoint" in manifest:
        _native_inventory(path, manifest["native_checkpoint"], hashes=True)
    rank, world = distributed.rank_world()
    payload = torch.load(path / ("state.pt" if world == 1 else f"rank-{rank:05d}.pt"),
                         map_location="cpu", weights_only=True)
    if any(payload[key] != manifest[key] for key in ("cycle", "cursor", "versions", "policy")):
        raise ValueError("checkpoint progress disagrees with manifest")
    if payload.get('data_progress') != manifest.get('data_progress'):
        raise ValueError('checkpoint stream progress disagrees with manifest')
    return payload


def restore_checkpoint(path, state, *, identity, native_checkpoint=None):
    payload = distributed.checked(lambda: read_checkpoint(path, identity=identity), "checkpoint files admission")
    distributed.checked(lambda: validate_rng(payload["rng"]), "checkpoint RNG admission")
    manifest = distributed.checked(lambda: read_manifest(path, identity=identity), "checkpoint owner metadata admission")
    native = manifest.get("native_checkpoint")
    if (native is None) != (native_checkpoint is None):
        raise ValueError("checkpoint native owner must be explicitly matched")
    if native is not None:
        # Stage *all* regular state and native layout before the first live
        # model/optimizer mutation. The engine must itself be newly built.
        progress = {key: payload[key] for key in ("cycle", "cursor", "versions")}
        distributed.checked(lambda: _stage_state(payload["state"], state, progress=progress), "checkpoint state pre-native admission")
        distributed.checked(lambda: native_checkpoint.validate_restore(Path(path)/"native", native["owner"]),
                            "native checkpoint pristine owner/layout admission")
        native_checkpoint.load(Path(path)/"native", native["owner"])
    unpack_state(payload["state"], state, progress={key:payload[key] for key in ("cycle", "cursor", "versions")})
    restore_rng(payload["rng"])
    return payload
