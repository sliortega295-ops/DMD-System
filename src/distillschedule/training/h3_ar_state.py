"""Native ZeRO-3 checkpoint component and complete-cycle W6 progress.

The real engine saves/loads its FP32 master shards and alternating Adam state.
This module serializes metadata only; it is not another optimizer backend.
"""
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

from distillschedule.core.contracts import require
from distillschedule.workloads.best_known.registry import file_sha256, fingerprint
from . import distributed

SCHEMA = "distillschedule.h3_ar_native_checkpoint.v1"
_ENGINE_CLIENT_METADATA = {"buffer_names", "param_shapes", "frozen_param_shapes", "shared_params",
                          "frozen_param_fragments", "global_samples", "ds_config", "ds_version"}


def _require(ok, message):
    require(ok, message, path="h3_ar.checkpoint")


class NativeCheckpoint:
    def __init__(self, session):
        self.session = session
        from distillschedule.strategies.h3_ar_session import deepspeed_config_receipt
        _require(session.engine.zero_load_from_fp32_weights() is True,
                 "native restart must restore original FP32 master weights, not reconstruct them from BF16")
        self.engine_config = deepspeed_config_receipt(session.engine.config)
        registered = [*session.adapter.role_parameters("student"), *session.adapter.role_parameters("fake_score"),
                      *session.discriminator.parameters()]
        self.permanent = {id(p) for p in registered}
        named = dict(session.engine.module.named_parameters())
        self.parameters = {n: p for n, p in named.items() if id(p) in self.permanent}
        _require(len(self.parameters) == len(self.permanent) == len(registered),
                 "permanent native trainables must all be registered once before checkpointing")
        self.layout = {name: dict(shape=list(getattr(p, "ds_shape", p.shape)), dtype=str(p.dtype))
                       for name, p in self.parameters.items()}
        self.partition_layout = self._partition_layout()
        _require([name for group in self.partition_layout for name, _ in group] != []
                 and {name for group in self.partition_layout for name, _ in group} == set(self.parameters)
                 and sum(map(len, self.partition_layout)) == len(self.parameters),
                 "native ordered FP32 optimizer partitions must own exactly both LoRAs and the discriminator")
        self.writers = [self._writer()] if distributed.rank_world()[1] == 1 else None
        self.progress = None

    def bind_writers(self):
        local = distributed.checked(self._writer, "native local checkpoint writer ownership")
        self.writers = distributed.gather(local)
        def validate():
            _require([row["rank"] for row in self.writers] == list(range(distributed.rank_world()[1]))
                     and all(row["save_model"] is True and row["save_optimizer"] is True for row in self.writers)
                     and len({row[k] for row in self.writers for k in ("model", "optimizer")}) == 2*len(self.writers),
                     "W6 requires actual all-rank native ZeRO3 model/master checkpoint writers")
        distributed.checked(validate, "native all-rank checkpoint writer ownership")

    def _partition_layout(self):
        # Exact installed DeepSpeed API: fp16_groups order, not dynamic
        # requires_grad flags or a sorted reconstruction of flat Adam shards.
        return [[[name, list(shape)] for name, shape in group.items()]
                for group in self.session.engine._get_zero_param_shapes()]

    def _writer(self):
        engine = self.session.engine
        _require(not engine.use_node_local_storage(), "native checkpoint output must be shared, not node-local")
        return dict(rank=distributed.rank_world()[0], save_model=engine.save_non_zero_checkpoint,
            save_optimizer=engine.save_zero_checkpoint,
            model=str(Path(engine._get_ckpt_name("NATIVE_ROOT", "state")).relative_to("NATIVE_ROOT")),
            optimizer=str(Path(engine._get_zero_ckpt_name("NATIVE_ROOT", "state")).relative_to("NATIVE_ROOT")))

    @contextmanager
    def registered_trainables(self):
        named = dict(self.session.engine.module.named_parameters())
        _require({id(p) for p in named.values()} >= self.permanent, "registered native trainable ownership changed")
        flags = {name: p.requires_grad for name, p in named.items()}
        _require(all(not flag or id(named[name]) in self.permanent for name, flag in flags.items()),
                 "frozen H3 backbone unexpectedly trainable")
        try:
            for parameter in self.parameters.values():
                parameter.requires_grad_(True)
            yield
        finally:
            for name, parameter in named.items():
                parameter.requires_grad_(flags[name])

    def validate_checkpoint_boundary(self):
        import torch
        s = self.session
        _require(s.engine.zero_optimization_stage() == 3 and s.trainer.generator_step == s.trainer.critic_step
                 and 1 <= s.trainer.generator_step <= 100 and s.engine.global_steps == 2*s.trainer.generator_step,
                 "native trainer/engine not at a complete G1/F1 boundary")
        _require(not getattr(s.engine, "_active_backward", False)
                 and (not callable(getattr(s.engine, "is_gradient_accumulation_boundary", None))
                      or s.engine.is_gradient_accumulation_boundary()), "native pending backward/accumulation")
        _require(all(p.grad is None or not bool(torch.count_nonzero(p.grad))
                     for p in s.engine.module.parameters()), "native unconsumed gradients at publication")
        pool = s.pool
        if pool is not None:
            _require(pool.selected is None and pool.rollout_plan is None and pool.group is None
                     and not pool.prefetched and not pool.prefetch_groups and not pool.immutable_entries
                     and not any(pool.consumer_queues) and not pool.pending
                     and all(not getattr(pool, name, {}) for name in ("sources", "entries", "routing", "last_consumed", "prefetch_stats"))
                     and all(not b.busy for group in pool.buffers.values() for b in group),
                     "native transient cache/prefetch/FIFO must drain before publication")
        _require(s.optimizer.active_role == "student" and s.optimizer.train_heads is False,
                 "native alternating optimizer role/head ownership is not a complete inactive-GAN boundary")

    def metadata(self):
        _require(self.progress is not None, "native checkpoint progress owner missing")
        _require(self.writers is not None, "native all-rank checkpoint writer ownership not bound")
        cycle = self.session.trainer.generator_step
        return dict(schema_version=SCHEMA, tag="state", zero_stage=3, world_size=distributed.rank_world()[1],
            layout=deepcopy(self.layout), partition_layout=deepcopy(self.partition_layout), writers=deepcopy(self.writers),
            engine_config=deepcopy(self.engine_config),
            client_state=dict(schema_version=SCHEMA, cycle=cycle,
                native_global_steps=2*cycle, input_sha256=self.progress.cursor.input_sha256,
                trainable_layout_sha256=fingerprint(dict(named=self.layout, partitions=self.partition_layout))))

    def validate_inventory(self, root, metadata):
        _require(isinstance(metadata, dict) and set(metadata) == {"schema_version", "tag", "zero_stage", "world_size", "layout", "partition_layout", "writers", "engine_config", "client_state"}
                 and metadata["schema_version"] == SCHEMA and metadata["tag"] == "state" and metadata["zero_stage"] == 3
                 and metadata["world_size"] == distributed.rank_world()[1] and metadata["layout"] == self.layout
                 and metadata["partition_layout"] == self.partition_layout == self._partition_layout()
                 and metadata["engine_config"] == self.engine_config
                 and metadata["writers"] == self.writers and self._writer() == self.writers[distributed.rank_world()[0]],
                 "native source/layout/topology checkpoint owner changed")
        names = {str(p.relative_to(root)) for p in Path(root).rglob("*.pt")}
        expected = {row[k] for row in self.writers for k in ("model", "optimizer")}
        _require(names == expected,
                 "native ZeRO3 optimizer/model shard inventory incomplete")
        client = metadata["client_state"]
        _require(isinstance(client, dict) and set(client) == {"schema_version", "cycle", "native_global_steps", "input_sha256", "trainable_layout_sha256"}
                 and client["schema_version"] == SCHEMA and type(client["cycle"]) is int and 1 <= client["cycle"] <= 100
                 and client["native_global_steps"] == 2*client["cycle"]
                 and client["trainable_layout_sha256"] == fingerprint(dict(named=self.layout, partitions=self.partition_layout))
                 and self.progress is not None and client["input_sha256"] == self.progress.cursor.input_sha256,
                 "native checkpoint client state/layout/input clock changed")
        self.audit_model_payloads(root, metadata)

    def audit_model_payloads(self, root, metadata):
        """Trusted native pickle; SHA admission precedes this pre-mutation audit.

        Under ZeRO3 these module tensors are partitioned placeholders, not a
        consolidated backbone. Inspect one model shard at a time; never gather.
        """
        import torch
        from distillschedule.strategies.h3_ar_session import deepspeed_config_receipt
        named = dict(self.session.engine.module.named_parameters())
        buffers = set(self.session.engine.module.state_dict()) - set(named)
        for path in sorted((Path(root)/"state").glob("*model_states.pt")):
            value = torch.load(path, map_location="cpu", weights_only=False)
            _require(set(value.get("module", {})) == set(self.parameters) | buffers
                     and all(isinstance(value["module"][name], torch.Tensor)
                             and str(value["module"][name].dtype) == self.layout[name]["dtype"] for name in self.parameters)
                     and not value.get("frozen_param_shapes") and not value.get("frozen_param_fragments")
                     and value.get("dp_world_size") == metadata["world_size"] and value.get("mp_world_size") == 1
                     and value.get("ds_version") == "0.18.4"
                     and deepspeed_config_receipt(value.get("ds_config")) == self.engine_config
                     and value.get("global_steps") == metadata["client_state"]["native_global_steps"]
                     and all(value.get(k) == v for k, v in metadata["client_state"].items())
                     and [[[name, list(shape)] for name, shape in group.items()]
                          for group in value.get("param_shapes", [])] == self.partition_layout,
                     "native module key/ordered master shape/client state mismatch; strict=False cannot hide missing trainables")
            del value

    def save(self, root):
        self.validate_checkpoint_boundary()
        metadata = self.metadata()
        with self.registered_trainables():
            result = self.session.engine.save_checkpoint(str(root), tag="state", client_state=metadata["client_state"],
                save_latest=False, exclude_frozen_parameters=True)
        distributed.checked(lambda: _require(result is not False, "native engine checkpoint save failed"), "native engine save result")
        # Every native writer has returned before inspecting its complete shared
        # inventory. Hash each writer's files once, not a model-sized CPU copy.
        distributed.gather(True)
        rank = distributed.rank_world()[0]
        def local_files():
            own = {self.writers[rank][key] for key in ("model", "optimizer")}
            _require(all((Path(root)/name).is_file() for name in own), "native writer did not produce its own completed shards")
            files = {}
            for path in sorted(Path(root).rglob("*")):
                name = str(path.relative_to(root))
                writer = next((r["rank"] for r in self.writers if name in (r["model"], r["optimizer"])), 0)
                if writer == rank and path.is_file():
                    files["native/"+name] = file_sha256(path)
            return files
        files = distributed.checked(local_files, "native completed local shard hashes")
        return dict(owner=metadata, files=files)

    def validate_restore(self, root, metadata):
        s = self.session
        _require(s.engine.global_steps == 0 and s.trainer.generator_step == s.trainer.critic_step == 0
                 and s.local_completed == 0, "native ZeRO3 restore requires a pristine new-process engine")
        self.validate_inventory(root, metadata)
        _require(metadata["client_state"]["cycle"] == self.progress.cursor.completed,
                 "native checkpoint and prepared input cursor differ")

    def load(self, root, metadata):
        self.validate_restore(root, metadata)
        with self.registered_trainables():
            loaded, client = self.session.engine.load_checkpoint(str(root), tag="state", load_module_strict=False,
                load_optimizer_states=True, load_lr_scheduler_states=True, load_module_only=False)
        # Pinned DeepSpeed0.18.4 returns its own non-excluded metadata alongside
        # user client_state (not just the supplied dictionary). Bind the user
        # fields exactly, and admit only the installed API's known extra keys.
        distributed.checked(lambda: _require(loaded is not None and isinstance(client, dict)
            and set(client) == set(metadata["client_state"]) | _ENGINE_CLIENT_METADATA
            and {k: client[k] for k in metadata["client_state"]} == metadata["client_state"]
            and not client["frozen_param_shapes"] and not client["frozen_param_fragments"]
            and [[[name, list(shape)] for name, shape in group.items()] for group in client["param_shapes"]] == self.partition_layout
            and self.session.engine.global_steps == client["native_global_steps"],
            "native engine did not restore exact client state/optimizer clock"), "native engine restore result")


class Progress:
    def __init__(self, session, cursor, native_checkpoint):
        self.session, self.cursor, self.native = session, cursor, native_checkpoint
        _require(native_checkpoint.progress is None, "native checkpoint already has a progress owner")
        native_checkpoint.progress = self

    def validate_checkpoint_boundary(self):
        self.native.validate_checkpoint_boundary()
        _require(self.cursor.pending is None and self.cursor.completed == self.session.trainer.generator_step,
                 "native data cursor has a pending or uncommitted input")

    def state_dict(self):
        s = self.session
        return dict(schema_version=SCHEMA, rank=s.rank, cycle=self.cursor.completed, cursor=self.cursor.completed,
            input_sha256=self.cursor.input_sha256, horizon=self.cursor.cycles, seed=self.cursor.seed,
            versions=s.versions(), native_global_steps=s.engine.global_steps, layout=deepcopy(self.native.layout),
            requires_grad={n: p.requires_grad for n, p in s.engine.module.named_parameters()},
            module_training={n: m.training for n, m in s.engine.module.named_modules()},
            optimizer_policy=dict(active_role=s.optimizer.active_role, train_heads=s.optimizer.train_heads),
            transient_state="RECONSTRUCT_COLD_NOT_PERSISTED")

    def validate_state_dict(self, value):
        s = self.session
        cycle = value.get("cycle")
        _require(set(value) == {"schema_version", "rank", "cycle", "cursor", "input_sha256", "horizon", "seed", "versions",
                  "native_global_steps", "layout", "requires_grad", "module_training", "optimizer_policy", "transient_state"}
                 and value["schema_version"] == SCHEMA and value["rank"] == s.rank
                 and type(cycle) is int and 1 <= cycle <= self.cursor.cycles
                 and value["cursor"] == cycle and value["versions"] == dict(student=cycle, fake=cycle, ema=0)
                 and all(type(v) is int for v in value["versions"].values())
                 and value["native_global_steps"] == 2*cycle and value["layout"] == self.native.layout
                 and value["input_sha256"] == self.cursor.input_sha256 and value["horizon"] == self.cursor.cycles
                 and value["seed"] == self.cursor.seed and value["optimizer_policy"] == dict(active_role="student", train_heads=False)
                 and value["transient_state"] == "RECONSTRUCT_COLD_NOT_PERSISTED", "native progress/source/layout identity changed")
        named = dict(s.engine.module.named_parameters())
        active = {id(p) for p in s.adapter.role_parameters("student")}
        _require(value["requires_grad"].keys() == named.keys()
                 and all(type(v) is bool and v == (id(named[n]) in active)
                         for n, v in value["requires_grad"].items())
                 and value["module_training"].keys() == dict(s.engine.module.named_modules()).keys()
                 and all(type(v) is bool for v in value["module_training"].values()),
                 "native trainability/module ownership changed")

    def validate_checkpoint_progress(self, value, progress):
        self.validate_state_dict(value)
        _require(progress is not None and all(value[k] == progress[k] for k in ("cycle", "cursor", "versions")),
                 "native progress disagrees with common publication")

    def load_state_dict(self, value):
        self.validate_state_dict(value)
        s = self.session
        _require(s.engine.global_steps == value["native_global_steps"], "engine must restore original Adam before trainer clocks")
        s.trainer.generator_step = s.trainer.critic_step = value["cycle"]
        s.adapter.select("student", trainable=True)
        s.discriminator.requires_grad_(False)
        for name, parameter in s.engine.module.named_parameters():
            parameter.requires_grad_(value["requires_grad"][name])
        for name, module in s.engine.module.named_modules():
            module.training = value["module_training"][name]
        s.optimizer.active_role, s.optimizer.train_heads = "student", False
        # The constructor already validated the next fresh record, without
        # committing it. It is reconstructed process-local input, not FIFO state
        # restored from the checkpoint and not a consumed/replayed update.
        pending = self.cursor.pending
        self.cursor.completed = value["cursor"]
        self.cursor.pending = pending if pending == value["cursor"]+1 else None
        s.rebase_segment(value["cycle"])
