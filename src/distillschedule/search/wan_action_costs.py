"""Component identities for the native, generated-order Wan execution path.

No timings are embedded here. Costs are keyed by actual work, local kernel,
precision, placement and warm state, not a whole candidate or an occurrence.
The first model is serial world16: it never invents inter-action overlap.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from distillschedule.core.graph import Dependency, ExecutionGraph
from distillschedule.runtime.cycle_plan import build_cycle_graph, rollout_window_order
from distillschedule.workloads.best_known.registry import load_profile
from .contracts import require
from .cost_model import ComponentContract
from .wan_implementation import PRECISION, WAN, WanImplementation

ADAPTER = "wan_prepared_actions_v2"
RANKS = tuple(range(16))
FAMILIES = {"g": "wan.generator_update", "r": "wan.critic_input_rollout",
            "c": "wan.critic_update", "e": "wan.generator_ema"}
MEMORY_DEFINITION = "PEAK_MINUS_STATIC_MINUS_ENTRY_RETAINED_INPUTS; OUTPUT_LIFETIME_STARTS_AT_PRODUCER_EXIT"
TIMING_DEFINITION = "SYNCHRONIZED_NATIVE_ACTION_CALIBRATION_NOT_FORMAL_CYCLE_TIMING"


def action_graph(window: int | None = None) -> ExecutionGraph:
    initial = load_profile(WAN)["contract"]["initial_versions"]
    versions, operations, edges = dict(initial), [], []
    for cycle in (1, 2):
        graph = build_cycle_graph(source_steps=(10, 11, 12, 13, 14),
                                 initial_versions=versions, ema_after=(0, 1, 2, 3, 4))
        prefix = f"cycle{cycle}."
        order = rollout_window_order(graph, max_live_rollouts=window) if window is not None else graph.topological_order()
        by_id = {op.operation_id: op for op in graph.operations}
        for key in order:
            op = by_id[key]
            operations.append(replace(op, operation_id=prefix+key,
                metadata={**op.metadata, "cycle": cycle, "action": key,
                          "inputs": tuple(prefix+x for x in op.metadata["inputs"])}))
        edges.extend(replace(d, source=prefix+d.source, target=prefix+d.target) for d in graph.dependencies)
        if window is not None:
            edges.extend(Dependency(prefix+a, prefix+b, "selected_native_order") for a,b in zip(order,order[1:]))
        versions = graph.metadata["final_versions"]
    edges.append(Dependency("cycle1.e4", "cycle2.g", "same_process_cycle_boundary"))
    return ExecutionGraph(tuple(operations), tuple(edges), {
        "adapter": ADAPTER, "initial_versions": initial, "final_versions": versions,
        "cycle_ends": {"1":"cycle1.e4", "2":"cycle2.e4"},
        "native_worker": "scripts/distillschedule/probe_wan13b_two_cycle_timing_world16.py",
        "scope": "SERIAL_NATIVE_ACTIONS; INTERNAL_COMPUTE_COMM_OVERLAP_IS_INSIDE_MEASURED_COST",
    })


def component_contract(operation, implementation: WanImplementation,
                       work: Mapping[str, Any]) -> ComponentContract:
    require(implementation.rollout_window is not None, "action costs require generated schedules")
    require(work and work.get("physical_batch") == 1 and work.get("logical_batch") == 16,
            "missing actual B1/globalB16 work contract")
    action, cycle = operation.metadata.get("action", ""), operation.metadata.get("cycle")
    require(action[:1] in FAMILIES and cycle in (1, 2), "not a bound native Wan action")
    kind = action[0]
    local = {"native_family": FAMILIES[kind]}
    if implementation.candidate_bundle.enabled:
        local.update(implementation.candidate_bundle.component_identity())
    if kind in "grc":
        local["modulation_chunk_tokens"] = implementation.chunk_tokens
    if kind in "gr":
        local["generator_nograd_lora_epilogue"] = implementation.lora_epilogue
    return ComponentContract(
        family=FAMILIES[kind], work=work,
        precision={"execution_precision_sha256":PRECISION.digest},
        placement={"world_size":16, "placement":implementation.placement,
                   "gradient_convention":"FP32_SUM_POST_DIVIDE_16",
                   "parameters":"STATIC_RESIDENT" if implementation.placement == "resident" else "FULL_SHARD_RESHARD_TRUE",
                   "nccl_algorithm":"Tree" if implementation.placement == "resident" else "AUTO"},
        implementation=local,
        context={"warm_state":"RESTORED_FIRST_CYCLE" if cycle == 1 else "AFTER_ONE_NATIVE_CYCLE",
                 "concurrency":"EXCLUSIVE_WORLD16_ACTION_ENVELOPE",
                 "retained_inputs":"SEPARATE_MEASURED_STORAGE_LIFETIMES_MAX_FIVE_B1",
                 "persistent_generator_unshard":False,
                 "allocation_claim":"PREDICTION_REQUIRES_NATIVE_CAPACITY_CONFIRMATION"},
    )


def trace_work_signatures(trace, rollout_contract: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Use already-validated manifest metadata, never load/copy tensor values.

    All sixteen rank layouts enter the work identity. Tensor content and sample
    ownership remain bound by the asset/semantic identity and native receipts.
    Reuse across distinct samples is legal only when their physical work agrees.
    """
    from runtime.static_world16_trace import _generator_keys, _critic_keys
    require(trace.variant == "no-gan" and tuple(trace.logical_steps) == (10,11,12,13,14),
            "component calibration requires the exact no-GAN trace")
    require(set(trace._control) == set(RANKS), "all rank layouts required")
    inventories = {row["operation"]:row["inventory"] for row in rollout_contract}
    def layouts(keys):
        out = []
        for rank in RANKS:
            tensors = trace._control[rank].manifest["tensors"]
            row = {}
            for name,key in sorted(keys.items()):
                descriptor = tensors[key]
                require("shape" in descriptor and "dtype" in descriptor, "trace tensor layout missing")
                row[name] = {"shape":descriptor["shape"], "dtype":descriptor["dtype"],
                             "staging":"CLONE_THEN_NATIVE_DEVICE_COPY"}
            out.append(row)
        return out
    common = {"physical_batch":1, "logical_batch":16, "output_shape":[1,16,9,48,80],
              "output_dtype":"torch.bfloat16", "score_packing":False}
    g_layout = layouts(_generator_keys(10,gan=False))
    work = {"g":{**common,"input_layouts":g_layout,
                 "dit_inventory":inventories["step/10/generator/rollout"],
                 "score_forwards":3,"backward_graphs":1,"optimizer_updates":1}}
    for i,step in enumerate(trace.logical_steps):
        row = layouts(_critic_keys(step,gan=False))
        work[f"r{i}"] = {**common,"input_layouts":row,
            "dit_inventory":inventories[f"step/{step}/critic/rollout"], "grad_enabled":False}
        work[f"c{i}"] = {**common,"input_layouts":row,"input_residency":"RETAINED_FROM_ROLLOUT",
                          "score_forwards":1,"backward_graphs":1,"optimizer_updates":1}
        work[f"e{i}"] = {"physical_batch":1,"logical_batch":16,"role":"generator",
                          "ema_updates":1,"ema_implementation":"reference",
                          "parameter_layout":"BOUND_MODEL_TRAINABLES",
                          "device":"cuda"}
    return work
