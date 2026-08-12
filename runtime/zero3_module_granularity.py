"""Read-only mirror of DeepSpeed's ZeRO-3 module granularity calculation."""
from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from typing import Iterable

import torch


@dataclass(frozen=True)
class ModuleGranularityRecord:
    name: str
    module_type: str
    parameter_numel: int
    descendant_layer_count: int
    granularity: int
    container_excluded: bool
    oversized_excluded: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def census_zero3_module_granularity(
    module: torch.nn.Module,
    *,
    max_leaf_numel: int = 1_000_000_000,
) -> list[ModuleGranularityRecord]:
    """Match DeepSpeed 0.18.4's threshold census without mutating the model.

    DeepSpeed uses ds_numel after partition conversion; before conversion it
    equals the logical parameter numel used here. Container modules and modules
    above max_leaf_numel are assigned sys.maxsize.
    """

    names = {id(value): name or "<root>" for name, value in module.named_modules()}
    records: list[ModuleGranularityRecord] = []

    def visit(current: torch.nn.Module) -> tuple[int, int]:
        recursive_parameters = tuple(current.parameters())
        if not recursive_parameters:
            return 0, 0

        layer_count = 0
        parameter_numel = sum(
            int(parameter.numel())
            for parameter in current.parameters(recurse=False)
        )
        children = tuple(current.children())
        if not children:
            return 1, parameter_numel

        for child in children:
            child_layers, child_numel = visit(child)
            layer_count += child_layers
            parameter_numel += child_numel

        container_excluded = (
            current.__class__.__name__ in torch.nn.modules.container.__all__
        )
        oversized_excluded = parameter_numel > max_leaf_numel
        if container_excluded:
            granularity = sys.maxsize
        else:
            layer_count += 1
            granularity = (
                parameter_numel // layer_count
                if not oversized_excluded
                else sys.maxsize
            )
            records.append(
                ModuleGranularityRecord(
                    name=names[id(current)],
                    module_type=current.__class__.__name__,
                    parameter_numel=parameter_numel,
                    descendant_layer_count=layer_count,
                    granularity=granularity,
                    container_excluded=False,
                    oversized_excluded=oversized_excluded,
                )
            )
        return layer_count, parameter_numel

    visit(module)
    return records


def selected_preorder_leaf_records(
    records: Iterable[ModuleGranularityRecord],
    *,
    threshold: int,
) -> list[ModuleGranularityRecord]:
    """Approximate DeepSpeed's parent-first threshold selection by module name."""

    if threshold <= 0:
        return []
    eligible = {
        record.name: record
        for record in records
        if record.granularity <= threshold
    }
    selected: list[ModuleGranularityRecord] = []
    selected_names: set[str] = set()
    for name in sorted(eligible, key=lambda value: (value.count("."), value)):
        parts = name.split(".")
        if any(".".join(parts[:depth]) in selected_names for depth in range(1, len(parts))):
            continue
        selected.append(eligible[name])
        selected_names.add(name)
    return selected
