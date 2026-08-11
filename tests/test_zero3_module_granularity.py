import sys

import torch

from runtime.zero3_module_granularity import (
    census_zero3_module_granularity,
    selected_preorder_leaf_records,
)


class _Branch(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.left = torch.nn.Linear(4, 4, bias=False)
        self.right = torch.nn.Linear(4, 4, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.left(value) + self.right(value)


def test_granularity_matches_parameter_over_layer_count() -> None:
    model = _Branch()
    records = census_zero3_module_granularity(model)
    root = next(record for record in records if record.name == "<root>")
    assert root.parameter_numel == 32
    assert root.descendant_layer_count == 3
    assert root.granularity == 10
    assert root.granularity < sys.maxsize


def test_parent_first_selection_does_not_duplicate_descendants() -> None:
    model = torch.nn.Module()
    model.branch = _Branch()
    records = census_zero3_module_granularity(model)
    selected = selected_preorder_leaf_records(records, threshold=10)
    names = {record.name for record in selected}
    assert "branch" in names
    assert not any(name.startswith("branch.") for name in names)
