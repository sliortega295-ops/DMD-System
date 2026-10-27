"""Opt-in finite full-C51 lifecycle over the original persistent native worker.

The worker still installs every C51 mechanism and owns all math, batching,
reductions, Adam and operation-keyed RNG. This adapter owns physical input
progress, legal publication drains, current-file assets and same-topology
restart. SOURCE/CPU evidence is not native CUDA numerical qualification.
"""
from copy import deepcopy
from pathlib import Path
import random
import sys

from distillschedule.data.livewan_inputs import C51_KIND, C51_SCHEMA, CONTRACT, _rows
from distillschedule.runtime.livewan_c51_bundle import source_template, validate_plan, receipt_contract
from distillschedule.search.livewan_request import LiveWanRequest
from distillschedule.workloads.best_known.registry import file_sha256, fingerprint
from . import distributed
from .livewan_reference_assets import document, require
from .livewan_c51_assets import C51Assets, asset_config, admit_runtime, source_origins
from .livewan_c51_state import PoolInputs, Progress, WindowedRunaheadTransport, owner_state, physical_versions

SCHEMA = "distillschedule.livewan_c51_continuation.v1"
POLICY = "distillschedule.livewan_c51_checkpoint.v1"
CAPACITY = "distillschedule.livewan_c51_capacity.v1"
TOPOLOGY = dict(world_size=16, backend="nccl", node_groups=[list(range(8)), list(range(8, 16))])


def descriptor():
    return dict(schema_version=SCHEMA, bundle="c51_full_v1", source_template=receipt_contract(),
                native_math="_run_pipeline_8x8_continuous/_run_pipeline_8x8", topology=deepcopy(TOPOLOGY),
                resolution=[368, 640], frames=81, global_batch=16,
                generator_physical_batch=1, critic_physical_batch=2, waves=2,
                updates=dict(generator=1, critic=5, ema=0), sigma=CONTRACT["sigma"],
                rng="unchanged independent operation-keyed six streams; NOT sequential upstream RNG equivalence",
                scheduling="depth1 lookahead; suppress only until explicit checkpoint/final drain completes",
                scratch="two pinned host slots per role; streamer/KV/SAC/cast/shadow reconstructed, never checkpointed",
                numerical_status="REVIEW", gpu_execution="NOT_RUN", claimable_speedup=False)


def capacity_policy(value):
    from distillschedule.search.resources import validate_capacity_policy
    require(type(value) is dict and set(value) == {"schema_version", "gpu_metric", "memory_limits", "host_limits",
            "gpu_reserve_bytes", "host_reserve_bytes"} and value["schema_version"] == CAPACITY,
            "explicit C51 capacity/reserve policy required")
    validate_capacity_policy(value)
    return deepcopy(value)


def input_metadata(path, *, cycles, seed, source):
    path = Path(path)
    before, value = file_sha256(path), document(path)
    require(value.get("schema_version") == C51_SCHEMA and value.get("input_sha256") == fingerprint(
                {k: v for k, v in value.items() if k != "input_sha256"})
            and value.get("status") == "ADMITTED_FINITE_NATIVE_INPUTS" and value.get("contract") == CONTRACT
            and type(value.get("cycles")) is int and value["cycles"] == cycles and 2 <= cycles <= 100
            and type(value.get("seed")) is int and value["seed"] == seed,
            "C51 explicit finite fresh input schema/hash/horizon/seed/geometry changed")
    origin = value.get("source", {})
    require(origin.get("head") == source["head"] and origin.get("root") == source["root"]
            and origin.get("files") and all(source["files"].get(k) == sha for k, sha in origin["files"].items()),
            "C51 prepared original source differs")
    records = value.get("records")
    require(type(records) is list and len(records) == 80*cycles
            and all(r["dataset_index"] == i for i, r in enumerate(records))
            and all(len({r[k] for r in records}) == len(records) for k in ("path", "sha256", "record_id", "latent_sha256"))
            and value.get("rows") == _rows(records, cycles, seed), "C51 repeated/missing/reordered fresh records or controls")
    require(file_sha256(path) == before, "C51 prepared manifest changed during metadata admission")
    return value


def admit_training(config, task_value, assets_value, paths):
    require(LiveWanRequest.from_dict(task_value) == LiveWanRequest(), "C51 supports original 81 frames only")
    assets = asset_config(assets_value)
    inputs = input_metadata(config.data["prepared"], cycles=config.trainer["cycles"], seed=config.trainer["seed"],
                            source=assets["native"]["sources"]["livewan"])
    policy = capacity_policy(document(paths["capacity_policy"]))
    make_plan(assets, paths["output"])  # exact frozen mechanism admission before model construction
    return dict(task=task_value, selected=dict(preset="self_forcing.c51_stream.v1", bundle="c51_full_v1"),
                assets=assets, inputs=inputs, capacity_policy=policy, lifecycle=descriptor(),
                native_worker="distillschedule.training.livewan_c51_training:run_training",
                validation_scope="SOURCE_INPUT_ASSET_METADATA_ONLY; NATIVE_CUDA_NUMERICAL_AND_CAPACITY_NOT_RUN")


def source_files():
    root = Path(__file__).parents[1]
    paths = set(root.glob("runtime/livewan*.py")) | set(root.glob("training/livewan*.py"))
    paths |= {root/"data/livewan_inputs.py", root/"training/checkpoint.py", root/"training/native_state.py"}
    return {str(p.relative_to(root)): file_sha256(p) for p in sorted(paths)}


def training_identity(head, admitted, config):
    return dict(schema_version=SCHEMA, source_head=head, source_files=source_files(),
                assets_sha256=fingerprint(admitted["assets"]), input_sha256=admitted["inputs"]["input_sha256"],
                horizon=config.trainer["cycles"], seed=config.trainer["seed"], checkpoint_every=config.trainer["checkpoint_every"],
                topology=deepcopy(TOPOLOGY), capacity_policy=deepcopy(admitted["capacity_policy"]), contract=descriptor())


def checkpoint_policy(identity):
    return dict(schema_version=POLICY, identity_sha256=fingerprint(identity), boundary="drained_complete_G1_C5_EMA0",
                persistent="actual resident role trainables/buffers/modes/named Adam; per-rank RNG and dual-wave cursor",
                scheduler="NONE_original_fixed_AdamW_learning_rates", frozen="Teacher/base reconstructed from explicit asset inventory")


def validate_progress(header, *, identity):
    cycle = header.get("cycle")
    require(type(cycle) is int and 0 < cycle < identity["horizon"] and header.get("cursor") == 5*cycle
            and header.get("versions") == dict(generator=cycle, critic=5*cycle, ema=0)
            and all(type(v) is int for v in header["versions"].values())
            and header.get("policy") == checkpoint_policy(identity), "C51 restart must be an exact complete nonterminal joint cycle")


def validate_joint(rows, *, cycle, pairs):
    """Prove both physical pools completed the same real 80-row cohort."""
    require(type(rows) is list and len(rows) == 16, "C51 missing physical rank progress")
    for rank, row in enumerate(rows):
        require(row.get("rank") == rank and row.get("cycle") == cycle
                and row.get("versions") == physical_versions(rank, cycle)
                and row.get("slots") == list(pairs[rank % 8]) and row.get("quiescent") is True,
                "C51 joint rank/topology/clock/quiescence differs")
    for lane in range(8):
        require(rows[lane]["rows"] == rows[lane+8]["rows"] and len(rows[lane]["rows"]) == 10,
                "C51 G production is not matching Score dual-wave consumption")
    produced = [r for lane in range(8) for r in rows[lane]["rows"]]
    require(len({r["operation_key"] for r in produced}) == 80 and len({r["record_id"] for r in produced}) == 80,
            "C51 incomplete/replayed actual global B16 cohort")
    return dict(cycle=cycle, cursor=5*cycle, versions=dict(generator=cycle, critic=5*cycle, ema=0))


def make_plan(assets, output, *, identity=None, stop=None):
    from distillschedule.core.plan import ExecutionPlan
    plan = ExecutionPlan.from_mapping(source_template()).to_dict()
    plan["id"] = "livewan_c51_continuation_v1"
    for role, name in (("generator", "student"), ("score", "critic"), ("teacher", "teacher")):
        plan["workload"][role]["checkpoint"] = assets["native"]["models"][name]["root"]
    plan["evidence"] = dict(output_dir=str(output), continuation_schema=SCHEMA)
    if identity is not None:
        plan["evidence"]["continuation_identity"] = deepcopy(identity)
        plan["evidence"]["requested_stop_cycle"] = stop
    require(validate_plan(plan, mode="pipeline"), "C51 exact full mechanism plan required")
    return plan


class Continuation:
    """Private typed opt-in seam consumed by the existing worker setup/loop."""
    def __init__(self, *, config, admitted, identity, output, bootstrap, loader, header, capacity_check):
        self.config, self.admitted, self.identity, self.output = config, admitted, identity, Path(output)
        self.bootstrap, self.loader, self.header, self.capacity_check = bootstrap, loader, header, capacity_check
        self.inputs = loader.inputs
        self.start_cycle = header["cycle"] if header else 0
        self.end_cycle = config.trainer.get("stop_after_cycle", self.inputs.cycles)
        require(self.start_cycle < self.end_cycle <= self.inputs.cycles, "C51 restart/stop has no unfinished complete cycles")
        self.cycles_executed = self.end_cycle-self.start_cycle
        self.state, self.progress, self.metrics = None, None, []
        self.pairs = None
        self._loss_count = self._norm_count = self._backward_count = 0
        self._calls = {}
        self.mechanism_receipt = None

    def configure_sources(self):
        assets = self.admitted["assets"]
        source_origins(assets)
        sources = dict(livewan=assets["native"]["sources"]["livewan"], wan21=assets["wan21_overlay"])
        for project in ("wan21", "livewan"):
            if sources[project]["root"] not in sys.path:
                sys.path.insert(0, sources[project]["root"])
        value = dict(schema_version=SCHEMA, sources=deepcopy(sources))
        return dict(value, source_receipt_sha256=fingerprint(value))

    def bind(self, *, torch, rank, bundle, versions, pairs):
        from .checkpoint import restore_checkpoint
        self.pairs = tuple(tuple(p) for p in pairs)
        self.loader = PoolInputs(self.loader, rank=rank, pairs=pairs)
        self.progress = Progress(self.loader, versions, restore_cycle=self.start_cycle if self.header else None)
        self.state = distributed.checked(lambda: owner_state(bundle,
            source_root=self.admitted["assets"]["native"]["sources"]["livewan"]["root"], progress=self.progress),
            "C51 actual resident named role state")
        if self.header:
            restored = restore_checkpoint(self.config.trainer["resume_from"], self.state, identity=self.identity)
            validate_progress(restored, identity=self.identity)
        self.progress.restore_cycle = None
        distributed.checked(lambda: require(versions == physical_versions(rank, self.start_cycle),
            "C51 restored actual role clock differs"), "C51 restored role clock")
        return self.loader

    def validate_constructed(self):
        self.bootstrap.unchanged()
        source_origins(self.admitted["assets"])
        admit_runtime(self.admitted["assets"])

    def transport(self, **kwargs):
        return WindowedRunaheadTransport(start_cycle=self.start_cycle, **kwargs)

    def prime(self, *, torch, rank, trace, bundle, streamers, loader, device, transport, shadow, evidence, elision):
        """Reconstruct only the necessary current-C shadow / next real wave0."""
        from distillschedule.runtime import livewan_crossscale_worker as worker
        if rank >= 8:
            shadow.begin_source(version=5*self.start_cycle)
            return None
        row = trace.row(5*self.start_cycle, self.pairs[rank][0])
        value = worker.prepare_rollout(torch, row=row, streamer=streamers[0], student=bundle.student,
                                      loader=loader, device=device, elide_full_context_lens=elision)
        worker._record_output(torch, evidence, row=row, x0=value.x0, run_kind="continuation")
        transport.enqueue_send(step=5*self.start_cycle, wave=0, tensor=value.x0)
        evidence.transport_bytes += value.x0.numel()*value.x0.element_size()
        return value

    def cycle_complete(self, *, torch, rank, cycle, versions, bundle, transport, shadow, evidence, inventory, streamers, device):
        from .checkpoint import save_checkpoint
        from distillschedule.runtime.livewan_crossscale_worker import atomic_json
        distributed_local_rows = list(self.loader.pending)
        self.loader.complete(cycle, versions)
        import math
        losses = evidence.loss_fingerprints[self._loss_count:]
        norms = evidence.optimizer_diagnostics[self._norm_count:]
        require(len(losses) == (6 if rank < 8 else 10) and len(norms) == (1 if rank < 8 else 5)
                and all(v.get("finite") is True and all(math.isfinite(v[k]) for k in ("mean", "norm", "max_abs")) for v in losses)
                and all(math.isfinite(v["clip_input_norm"]) for v in norms), "C51 actual native loss/clip evidence missing/nonfinite")
        delta = {k: v-self._calls.get(k, 0) for k, v in inventory.counts.items()}
        rows = [self.inputs.row(5*(cycle-1)+s, slot) for s in range(5) for slot in self.pairs[rank % 8]]
        if rank < 8:
            require(delta.get("student/nograd/plain") == sum(1+5*r.new_blocks for r in rows)
                    and delta.get("student/grad/checkpoint") == 6,
                    "C51 row-derived original Generator call/recompute census changed")
        else:
            require(delta.get("critic/grad/checkpoint") == 5 and delta.get("teacher/nograd/plain") == 2
                    and evidence.physical_critic_backward_calls-self._backward_count == 5,
                    "C51 original physical B2 backward/Teacher census changed")
        self._loss_count, self._norm_count = len(evidence.loss_fingerprints), len(evidence.optimizer_diagnostics)
        self._backward_count, self._calls = evidence.physical_critic_backward_calls, dict(inventory.counts)
        cadence = self.config.trainer["checkpoint_every"]
        publish = cycle == self.end_cycle or bool(cadence and cycle % cadence == 0)
        self.progress.quiescent = False
        metric = dict(cycle=cycle, physical_versions=dict(versions), consumed=distributed_local_rows,
                      physical_call_inventory=dict(inventory.counts), logical_score_calls=dict(evidence.logical_score_calls),
                      losses=deepcopy(losses), optimizer_diagnostics=deepcopy(norms),
                      input_sha256=self.inputs.input_sha256, publication_boundary=publish)
        self.metrics.append(metric)
        if not publish:
            # Keep the existing per-complete-cycle capacity/OOM/reserve gate.
            # It is local here: a metadata barrier would suppress C51's depth1
            # overlap. Publication below exchanges any boundary failure before
            # checkpoint collectives and repeats this check after full drain.
            self.capacity_check()
            return
        # Before any next-cycle lookahead or shadow submission. Source futures
        # from the preceding cycle must already have matched their receivers.
        transport.drain_through(cycle)
        require(shadow.pending_source_work is None and shadow.pending_source_version is None,
                "C51 unmatched critic-shadow broadcast at publication")
        if rank < 8:
            loaded = shadow.events[-1]["version"] if shadow.events else 0
            require(loaded == 5*(cycle-1), "C51 wrong loaded Critic shadow version")
            bundle.generator_optimizer.zero_grad(set_to_none=True)
        else:
            require(bundle.critic_gradient_reducer is not None, "C51 actual async reducer missing")
            bundle.critic_gradient_reducer.zero_grad()  # retain persistent gradient views
        torch.cuda.synchronize(device)
        self.progress.quiescent = True
        from distillschedule.runtime.livewan_c51_bundle import continuation_receipt_errors
        require(callable(self.mechanism_receipt), "C51 actual installed mechanism observers missing")
        mechanism = self.mechanism_receipt()
        gate = {**mechanism, "output_fingerprints": list(evidence.output_fingerprints),
            "continuation": dict(schema_version=SCHEMA, start_cycle=self.start_cycle, completed_cycles=cycle,
                requested_stop_cycle=cycle, target_cycles=self.inputs.cycles, input_sha256=self.inputs.input_sha256,
                executed_cycles=cycle-self.start_cycle, source_reuse_across_cycles=False,
                numerical_status="REVIEW", physical_final_versions=dict(versions))}
        distributed.checked(lambda: require(not continuation_receipt_errors(gate, rank=rank, inputs=self.inputs, pairs=self.pairs),
            "C51 installed full-bundle runtime counters/source gate failed before checkpoint"), "C51 full-bundle publication gate")
        metric["mechanisms"] = deepcopy(mechanism)
        distributed.checked(lambda: [getattr(s, "validate_checkpoint_boundary", lambda: None)()
                                    for s in self.state.values()], "C51 actual local drained boundary")
        observation = dict(rank=rank, cycle=cycle, versions=dict(versions), slots=list(self.pairs[rank % 8]),
                           rows=distributed_local_rows, quiescent=True)
        joint = validate_joint(distributed.gather(observation), cycle=cycle, pairs=self.pairs)
        distributed.checked(self.capacity_check, "C51 complete-cycle capacity/reserve/OOM gate")
        if cadence and cycle % cadence == 0:
            save_checkpoint(self.output/"checkpoints"/f"cycle-{cycle:06d}", self.state, identity=self.identity,
                **joint, policy=checkpoint_policy(self.identity), pre_publication_guard=self.capacity_check)
        if cycle == self.end_cycle:
            distributed.checked(self.capacity_check, "C51 final completed-receipt capacity/reserve/OOM gate")
        atomic_json(self.output/"continuation"/f"rank-{rank:05d}.json", dict(schema_version=SCHEMA,
            status="COMPLETED" if cycle == self.inputs.cycles else "PARTIAL", identity=self.identity,
            progress=joint, physical_versions=dict(versions), metrics=self.metrics,
            native_cuda_numerical_qualification="NOT_RUN", claimable_speedup=False))
        # set_world resets all native KV cursors on the next real rollout.
        # Drop transient text/context references now; no recorded rollout or
        # recomputation survives the original helper's return or the checkpoint.
        for streamer in streamers:
            streamer.ctx = None
            streamer.ctx_lens = None


def run_training(config, *, admit):
    """Enter on every already-launched rank; reserve nothing and install nothing."""
    import torch
    from .checkpoint import read_manifest, read_checkpoint
    from .livewan_training import _reserve_output, _bind_devices, _capacity, _admit_model_layout
    from distillschedule.runtime import livewan_crossscale_worker as worker
    from distillschedule.search.resources import ResourceObserver
    with distributed.world16_reference_process_group("W4 full C51 continuation"):
        distributed.admit_run_controls(config)
        result = distributed.checked(admit, "C51 public source/assets/input/policy admission")
        _, source, paths, admitted = result
        require(not source["identity"]["dirty"], "C51 execution requires committed source, not a fake Git identity")
        identity = training_identity(source["identity"]["head"], admitted, config)
        require(all(row == identity for row in distributed.gather(identity)), "C51 rank source/input/policy identities differ")
        stop = config.trainer.get("stop_after_cycle", config.trainer["cycles"])
        require(all(row == stop for row in distributed.gather(stop)), "C51 rank stop controls differ")
        output = Path(paths["output"])
        _reserve_output(output)
        rank = distributed.rank_world()[0]
        runtime = distributed.checked(lambda: admit_runtime(admitted["assets"]), "C51 pinned runtime/precision/kernel roots")
        observer = ResourceObserver(output/"capacity"/f"rank-{rank:05d}.json", "full-C51-continuation", rank//8,
                                    capacity_policy=admitted["capacity_policy"])
        binding = _bind_devices(torch, observer, rank=rank)
        initial = distributed.checked(lambda: _capacity(torch, observer, admitted["capacity_policy"], rank=rank,
                                      binding=binding), "C51 pre-payload live capacity/reserve")
        check_capacity = lambda: _capacity(torch, observer, admitted["capacity_policy"], rank=rank,
                                            binding=binding, initial=initial)
        bootstrap = distributed.checked(lambda: C51Assets.load(admitted["assets"]), "C51 all frozen and Stage1 bytes")
        header = None
        if config.trainer.get("resume_from"):
            header = distributed.checked(lambda: read_manifest(config.trainer["resume_from"], identity=identity), "C51 all-rank restart metadata")
            validate_progress(header, identity=identity)
            distributed.checked(lambda: read_checkpoint(config.trainer["resume_from"], identity=identity), "C51 all shards/hashes before model mutation")
        loader = distributed.checked(lambda: worker.InputLoader(torch=torch, input_kind=C51_KIND,
            prepared=config.data["prepared"]), "C51 actual original InputLoader source/record admission")
        distributed.checked(lambda: require(loader.inputs.input_sha256 == identity["input_sha256"]
            and loader.inputs.cycles == config.trainer["cycles"]
            and loader.inputs._document["seed"] == config.trainer["seed"],
            "C51 actual InputLoader identity/horizon/seed differs"), "C51 actual InputLoader immutable identity")
        continuation = Continuation(config=config, admitted=admitted, identity=identity, output=output, bootstrap=bootstrap,
                                    loader=loader, header=header, capacity_check=check_capacity)
        continuation.configure_sources()
        layouts = distributed.checked(lambda: _admit_model_layout(bootstrap, worker), "C51 original complete parameter layout before model allocation")
        continuation.runtime, continuation.native_layouts, continuation.device_binding = runtime, layouts, binding
        distributed.checked(check_capacity, "C51 pre-model live capacity/reserve")
        # The model builder preserves its original construction-local LoRA seed.
        random.seed(config.trainer["seed"]+rank); torch.manual_seed(config.trainer["seed"]+rank)
        import numpy as np
        np.random.seed((config.trainer["seed"]+rank) % 2**32)
        plan = make_plan(admitted["assets"], output, identity=identity, stop=stop)
        plan_path = output/f"effective-training-plan.rank-{rank:05d}.json"
        worker.atomic_json(plan_path, plan)
        try:
            receipt = worker.run_worker(plan_path=plan_path, mode="pipeline", run_kind="continuation",
                                        repo_root=Path(__file__).parents[3], continuation=continuation)
        except BaseException as exc:
            # Preserve earlier complete checkpoints and progress separately;
            # never replace them with a misleading final/completed receipt.
            worker.atomic_json(output/"continuation"/f"rank-{rank:05d}.failed.json", dict(schema_version=SCHEMA,
                status="FAIL", identity=identity, rank=rank, error=type(exc).__name__+": "+str(exc),
                last_locally_completed_cycle=getattr(continuation.loader, "cycle", 0),
                native_cuda_numerical_qualification="NOT_RUN", claimable_speedup=False))
            raise
        require(receipt["continuation"]["completed_cycles"] == stop, "C51 requested finite segment not exhausted")
        return 0
