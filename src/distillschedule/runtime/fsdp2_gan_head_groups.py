"""Fail-closed ownership checks for separate, existing Helios GAN head units.

This helper never changes the module tree, parameters, forward calls or meshes.
Class objects are supplied by the already-loaded source-qualified model runtime.
"""
from __future__ import annotations

from collections import Counter
from typing import Any


GAN_HEAD_KEYS = ("5", "15", "25", "35")
GAN_HEAD_PATHS = tuple(f"gan_heads.{key}" for key in GAN_HEAD_KEYS) + ("gan_final_head",)


def validate_gan_head_fsdp_request(
    *, enabled: bool, variant: str, mesh_policy: str,
    rollout_batch_size: int, pack_real_score_pair: bool,
) -> None:
    if enabled and (
        variant != "gan" or mesh_policy != "world16-fsdp"
        or rollout_batch_size != 1 or pack_real_score_pair
    ):
        raise ValueError("separate GAN-head FSDP requires GAN world16-fsdp B1 with packing off")


def gan_head_fsdp_units(
    module: Any, *, head_type: type, container_type: type,
) -> tuple[tuple[str, Any], ...]:
    """Resolve exactly five disjoint head units before *any* FSDP wrapping.

    Non-deduplicated parameter traversal is intentional: aliases inside a head,
    between heads, or from a head into any other model subtree are forbidden.
    """
    if not getattr(module, "is_use_gan", False):
        raise ValueError("separate GAN-head FSDP requires the GAN Critic")
    container = getattr(module, "gan_heads", None)
    if type(container) is not container_type:
        raise ValueError("GAN-head container class identity changed")
    if tuple(container.keys()) != GAN_HEAD_KEYS:
        raise ValueError(f"GAN-head keys/order changed: expected {GAN_HEAD_KEYS}")
    units = tuple((f"gan_heads.{key}", container[key]) for key in GAN_HEAD_KEYS) + (
        ("gan_final_head", getattr(module, "gan_final_head", None)),
    )
    if len({id(head) for _, head in units}) != len(units):
        raise ValueError("GAN-head modules are missing or shared")
    occurrences = Counter(id(parameter) for _, parameter in module.named_parameters(remove_duplicate=False))
    registrations: dict[int, list[str]] = {}
    for name, child in module.named_modules(remove_duplicate=False):
        registrations.setdefault(id(child), []).append(name)
    selected: set[int] = set()
    for path, head in units:
        if type(head) is not head_type:
            raise ValueError(f"GAN-head class identity changed: {path}")
        names = registrations.get(id(head), [])
        if len(names) != 1 or not (names[0] == path or names[0].endswith("." + path)):
            raise ValueError(f"GAN-head module registration changed or shared: {path}")
        parameters = [parameter for _, parameter in head.named_parameters(remove_duplicate=False)]
        identifiers = [id(parameter) for parameter in parameters]
        if not identifiers:
            raise ValueError(f"GAN-head has no parameters: {path}")
        if len(set(identifiers)) != len(identifiers):
            raise ValueError(f"duplicate parameter references inside GAN-head: {path}")
        if selected.intersection(identifiers) or any(occurrences[value] != 1 for value in identifiers):
            raise ValueError(f"shared or unregistered GAN-head parameters: {path}")
        selected.update(identifiers)
    return units
