"""H3's generated Student/preparation/consumer actions, over the common IR.

The model is serial world16. Internal FA3/FSDP overlap stays inside measured
action envelopes; different windows do not manufacture pipeline overlap.
"""
from dataclasses import replace

from distillschedule.core.graph import Dependency, ExecutionGraph
from distillschedule.runtime.cycle_plan import build_cycle_graph, rollout_window_order
from distillschedule.workloads.best_known.registry import fingerprint
from .contracts import require
from .cost_model import ComponentContract
from .h3_implementation import H3

ADAPTER = "h3_prepared_actions_v2"
RANKS = tuple(range(16))
PRECISION = fingerprint(dict(base="BF16", trainables="UNCHANGED_PINNED_NATIVE_PEFT_FSDP_POLICY_NOT_A_NEW_DTYPE_PROBE",
                            attention="_flash_3_hub", sigma="APPROXIMATE_GRID1000"))
FAMILIES = dict(g="h3.student_update", r="h3.fake_input_preparation", c="h3.fake_update")


def action_graph(window=None):
    versions = dict(generator=0, critic=0, ema=0)
    ops, edges = [], []
    for cycle in (1, 2):
        graph = build_cycle_graph(source_steps=tuple(5*(cycle-1)+i for i in range(5)),
                                  initial_versions=versions, ema_after=())
        prefix = f"cycle{cycle}."
        order = rollout_window_order(graph, max_live_rollouts=window) if window is not None else graph.topological_order()
        by_id = {o.operation_id:o for o in graph.operations}
        ops.extend(replace(by_id[k], operation_id=prefix+k, metadata={**by_id[k].metadata,
            "cycle":cycle, "action":k, "inputs":tuple(prefix+x for x in by_id[k].metadata['inputs'])}) for k in order)
        edges.extend(replace(d, source=prefix+d.source, target=prefix+d.target) for d in graph.dependencies)
        if window is not None:
            edges.extend(Dependency(prefix+a,prefix+b,"selected_native_order") for a,b in zip(order,order[1:]))
        versions = graph.metadata['final_versions']
    edges.append(Dependency("cycle1.c4", "cycle2.g", "same_process_cycle_boundary"))
    return ExecutionGraph(tuple(ops),tuple(edges),dict(adapter=ADAPTER,
        initial_versions=dict(generator=0,critic=0,ema=0),final_versions=versions,
        cycle_ends={"1":"cycle1.c4","2":"cycle2.c4"},
        scope="SERIAL_WORLD16; INTERNAL_OVERLAP_INSIDE_MEASURED_ACTION"))


def component_contract(op, impl, work):
    require(impl.rollout_window is not None and work, "H3 action/work identity missing")
    kind = op.metadata['action'][0]
    return ComponentContract(FAMILIES[kind],work,{"execution_precision_sha256":PRECISION},
        dict(world_size=16, placement="world16-fsdp2", shared_backbone=True, reshard_after_forward=True),
        dict(native_family=FAMILIES[kind], lora_epilogue_block_n=impl.lora_epilogue_block_n,
             **({"fa3_cpu_staged_depth":impl.prefetch_depth} if kind in 'gc' else {})),
        dict(warm_state="SETUP_FIRST_CYCLE" if op.metadata['cycle']==1 else "AFTER_ONE_NATIVE_CYCLE",
             concurrency="EXCLUSIVE_WORLD16_ACTION_ENVELOPE", fa3_storage="cpu_staged",
             boundary_offload="none", retained_inputs="MEASURED_LAST_CONSUMER_STORAGE_LIFETIMES",
             memory="CONSERVATIVE_ALLOCATION_ENVELOPE_NATIVE_CAPACITY_CONFIRMATION_REQUIRED"))
