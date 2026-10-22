"""Finite, source-inventoried input for the original Grid1000 data owner.

Only deterministic cached-condition metadata is admitted in v1. This neither
converts raw media nor substitutes a trace for the upstream LatentDataset.
Metadata admission intentionally imports no Torch/model/CUDA dependency.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

from distillschedule.workloads.best_known.registry import file_sha256, fingerprint

SCHEMA = "original_h3_grid_latent_dataset.v1"
SAMPLER = {"class": "torch.utils.data.DistributedSampler", "num_replicas": 16,
           "shuffle": True, "drop_last": True, "seed": 0, "epoch": 0}


def require(condition, message):
    if not condition:
        raise ValueError("original H3 Grid training: " + message)


def document(path):
    path = Path(path).resolve()
    require(path.is_file(), f"missing JSON file: {path}")
    value = json.loads(path.read_text())
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def data_contract(config):
    data = config.get("data", {})
    require(set(data) == {"train"}, "v1 supports only an explicit finite training dataset")
    train = data["train"]
    require(isinstance(train, dict) and set(train) == {
        "name", "data_path", "batch_size", "dataset_repeat", "shuffle", "drop_last", "num_workers", "pin_memory"},
        "explicit original latent_dataset fields required; transforms/defaults are not inferred")
    require(train["name"] == "latent_dataset" and type(train["batch_size"]) is int and train["batch_size"] == 1
            and type(train["dataset_repeat"]) is int and train["dataset_repeat"] == 1
            and train["shuffle"] is True and train["drop_last"] is True
            and type(train["num_workers"]) is int and train["num_workers"] == 0
            and type(train["pin_memory"]) is bool,
            "v1 requires original B1/world16 shuffle/drop_last, dataset_repeat=1, workers=0 and no transforms")
    path = Path(train["data_path"]).expanduser()
    require(path.is_absolute() and path.suffix == ".jsonl" and path.is_file(),
            "data_path must be one existing absolute JSONL metadata file")
    require(not (path.parent / "negative_condition.pt").exists(),
            "implicit upstream negative_condition.pt discovery is outside v1")
    return train, path.resolve()


def _inventory(config_path, *, cycles):
    config_path = Path(config_path).resolve()
    config = document(config_path)
    train, metadata = data_contract(config)
    require(type(cycles) is int and 2 <= cycles <= 64
            and config["training"]["max_train_iters"] == cycles,
            "input inventory target must match the fixed original scheduler horizon (2..64)")
    rows = []
    for line in metadata.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        require(isinstance(record, dict) and set(record) == {"id", "condition_path"}
                and isinstance(record["id"], str) and bool(record["id"])
                and isinstance(record["condition_path"], str) and bool(record["condition_path"]),
                "v1 rows require exactly id/condition_path; raw media, transforms and traces are not supported")
        payload = (metadata.parent / record["condition_path"]).resolve()
        require(payload.is_file(), f"missing cached condition: {payload}")
        rows.append({"record": record, "condition_path": str(payload), "sha256": file_sha256(payload)})
    require(len(rows) >= 96 * cycles and len(rows) % 16 == 0,
            "fresh input requires at least 96N genuine rows, divisible by world16; no sampler padding/replay")
    require(all(len({r[key] for r in rows}) == len(rows) for key in ("condition_path", "sha256"))
            and len({r["record"]["id"] for r in rows}) == len(rows),
            "distinct sample ids, paths and cached payload identities required; fixed-trace duplicates rejected")
    value = {"schema_version": SCHEMA, "native_config_sha256": file_sha256(config_path),
             "cycles": cycles, "metadata": {"path": str(metadata), "sha256": file_sha256(metadata)},
             "data_config": train, "sampler": dict(SAMPLER), "rows": rows}
    return {**value, "inventory_sha256": fingerprint(value)}


def prepare_inputs(config_path, output):
    """Inventory supplied cached payloads, without copying or generating data."""
    config = document(config_path)
    value = _inventory(config_path, cycles=config["training"]["max_train_iters"])
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return value


def load_inputs(path, *, config_path, cycles):
    value = document(path)
    require(value == _inventory(config_path, cycles=cycles),
            "native config, input inventory, cached payloads, sampler or fixed target changed")
    return value


def validate_payloads(inventory):
    """Worker-only semantic qualification using the actual upstream dataset.

    A renamed/copied torch archive may hash differently despite containing the
    same fixed trace tensors. Check original loaded conditions, not filenames.
    This is a finite cached input contract, not a proof of raw-data provenance.
    """
    import torch
    from lightx2v_train.data.video_dataset import LatentDataset
    dataset = LatentDataset(inventory["metadata"]["path"])
    require(len(dataset) == len(inventory["rows"]) and dataset.negative_condition is None,
            "actual original cached dataset inventory changed")
    identities = []
    for index, expected in enumerate(inventory["rows"]):
        require(file_sha256(Path(expected["condition_path"])) == expected["sha256"], "cached condition changed before model construction")
        sample = dataset[index]
        condition = sample["conditioning"].get("positive")
        require(sample["meta"].get("id") == expected["record"]["id"] and not sample["inputs"]
                and isinstance(condition, dict) and set(condition) == {"prompt_embeds", "text_token_tags"},
                "original H3 cached text-only condition required")
        embeds, tags = condition["prompt_embeds"], condition["text_token_tags"]
        require(isinstance(embeds, torch.Tensor) and embeds.ndim == 2 and embeds.shape[0] > 0
                and embeds.shape[1] > 0 and embeds.dtype in (torch.bfloat16, torch.float32)
                and bool(torch.isfinite(embeds).all()) and isinstance(tags, torch.Tensor)
                and tags.ndim == 1 and tags.dtype == torch.int64 and len(tags) == len(embeds)
                and bool((tags == 1).all()), "original H3 cached condition shape/dtype/text-only gate failed")
        digest = hashlib.sha256()
        for tensor in (embeds, tags):
            digest.update(str((tuple(tensor.shape), str(tensor.dtype))).encode())
            digest.update(tensor.contiguous().view(torch.uint8).numpy().tobytes())
        identities.append(digest.hexdigest())
    require(len(set(identities)) == len(identities), "duplicate cached condition tensors are not fresh rows")
    return fingerprint(identities)


class NativeLatentCursor:
    """Seek the actual upstream loader without consuming restored model RNG.

    The fresh iterator retains upstream's one base-seed RNG draw. Only restart
    reconstruction/seek is isolated, including workers=0's iterator seed draw.
    v1 consumes a single epoch of unique data; it never enters replay/padding.
    """
    def __init__(self, loader, inventory, *, cycle, resume):
        from torch.utils.data import DistributedSampler, default_collate
        from lightx2v_train.data.video_dataset import LatentDataset
        from .checkpoint import rng_state, restore_rng
        require(type(cycle) is int and 0 <= cycle < inventory["cycles"], "invalid native data cursor cycle")
        sampler = loader.sampler
        dataset = loader.dataset
        require(type(dataset) is LatentDataset and type(sampler) is DistributedSampler
                and loader.batch_size == 1 and loader.num_workers == 0 and not loader.drop_last
                and loader.collate_fn is default_collate and dataset.dataset_repeat == 1
                and dataset.max_samples is None and dataset.negative_condition is None,
                "actual original deterministic LatentDataset/DataLoader owners required")
        require(sampler.num_replicas == 16 and sampler.shuffle is True and sampler.drop_last is True
                and sampler.seed == 0 and 0 <= sampler.rank < 16,
                "actual original world16 DistributedSampler identity changed")
        expected = inventory["rows"]
        require(len(dataset) == len(expected) and all(
            sample["type"] == "metadata" and sample["row"] == row["record"]
            and (Path(sample["base_dir"]) / sample["row"]["condition_path"]).resolve() == Path(row["condition_path"])
            for sample, row in zip(dataset.samples, expected, strict=True)),
            "actual original dataset inventory/order changed")
        self.inventory, self.loader = inventory, loader
        self.cursor, self.epoch = 0, 0
        self.limit = 6 * inventory["cycles"]
        sampler.set_epoch(0)
        self.order = list(sampler)
        require(len(self.order) >= self.limit and len(set(self.order)) == len(self.order),
                "not enough unique rank-local native samples")
        saved = rng_state() if resume else None
        try:
            self.iterator = iter(loader)
            for _ in range(6 * cycle):
                next(self)
        finally:
            if saved is not None:
                restore_rng(saved)

    def __iter__(self):
        return self

    def __next__(self):
        if self.cursor >= self.limit:
            raise StopIteration
        expected = self.inventory["rows"][self.order[self.cursor]]
        require(file_sha256(Path(expected["condition_path"])) == expected["sha256"],
                "cached condition changed before native next-sample consumption")
        sample = next(self.iterator)
        require(sample["meta"]["id"] == [expected["record"]["id"]]
                and sample["meta"]["condition_path"] == [expected["condition_path"]],
                "original dataloader returned a different next sample")
        self.cursor += 1
        return sample

    def validate_cycle(self, cycle):
        require(self.epoch == self.loader.sampler.epoch == 0 and self.cursor == 6 * cycle,
                "native sampler/next-sample cursor is not a complete G1/F5 boundary")
