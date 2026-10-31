"""Explicit workload conditions; query wording is never an observed fact."""
import re
from dmd_perf import require


def normalize_context(context):
    """Separate explicit hardware facts; a partial model name never invents capacity.

    This parses supplied text only, not the machine. Structured fields are preferred.
    Unknown hardware labels remain opaque; ambiguous labels infer no GPU family.
    """
    normalized = dict(context)
    # Normalize explicit values only. Query text never becomes an observed fact.
    aliases = {"grad_mode": {"no-grad": "no_grad", "nograd": "no_grad", "with_grad": "grad_enabled",
                            "requires_grad": "grad_enabled", "grad": "grad_enabled"},
               "base_dtype": {"bf16": "bfloat16", "fp32": "float32"},
               "dtype": {"bf16": "bfloat16", "fp32": "float32"},
               "lora_update_dtype": {"bf16": "bfloat16", "fp32": "float32"}}
    for key, mapping in aliases.items():
        if isinstance(normalized.get(key), list):
            normalized[key] = [mapping.get(v.lower(), v) if isinstance(v, str) else v for v in normalized[key]]
        if isinstance(normalized.get(key), str):
            normalized[key] = mapping.get(normalized[key].lower(), normalized[key])
    topology_keys = ("nodes", "gpus_per_node", "physical_gpus")
    for key in topology_keys:
        value = normalized.get(key)
        require(value is None or type(value) is int and value >= (1 if key == "nodes" else 0),
                "hardware topology requires positive nodes and nonnegative integer GPU allocations: " + key)
    if all(normalized.get(k) is not None for k in topology_keys):
        require(normalized["nodes"] * normalized["gpus_per_node"] == normalized["physical_gpus"],
                "contradictory hardware allocation topology")
    if isinstance(normalized.get("gpu_family"), str):
        normalized["gpu_family"] = normalized["gpu_family"].upper()
    label = normalized.get("hardware")
    if not isinstance(label, str):
        return normalized
    families = set(re.findall(r"\b(?:A100|H100|H20|B300)\b", label.upper()))
    if len(families) != 1:
        return normalized
    family = next(iter(families))
    parsed = {"gpu_family": family}
    capacities = set(re.findall(r"\b([0-9]+)\s*GB\b", label.upper()))
    if len(capacities) == 1:
        parsed["gpu_memory_gb"] = int(next(iter(capacities)))
    topology = re.search(r"\b([0-9]+)\s*[x×]\s*([0-9]+)\s+(?:NVIDIA\s+)?" + family + r"\b", label, re.I)
    count = re.search(r"\b([0-9]+)\s*[x×]\s*(?:NVIDIA\s+)?" + family + r"\b", label, re.I)
    if topology:
        parsed.update(nodes=int(topology[1]), gpus_per_node=int(topology[2]),
                      physical_gpus=int(topology[1]) * int(topology[2]))
    elif count:
        parsed["physical_gpus"] = int(count[1])
    for key, value in parsed.items():
        require(normalized.get(key) in (None, value), "contradictory hardware field: " + key)
        normalized[key] = value
    if all(normalized.get(k) is not None for k in topology_keys):
        require(normalized["nodes"] * normalized["gpus_per_node"] == normalized["physical_gpus"],
                "contradictory hardware allocation topology")
    del normalized["hardware"]
    return normalized


def condition_match(conditions, context):
    conditions, context = normalize_context(conditions), normalize_context(context)
    unknown, conflict, matched = [], [], []
    for key, expected in conditions.items():
        if expected is None or context.get(key) is None:
            unknown.append(key)
        elif (context[key] in expected if isinstance(expected, list) else context[key] == expected):
            matched.append(key)
        else:
            conflict.append(key)
    return {"matched": matched, "conflicting": conflict, "unknown": unknown}
