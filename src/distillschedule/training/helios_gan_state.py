"""Serialization-only ownership of the original low-VRAM GAN critic phases.

The original trainer binds base/extra names before updates, and enables both
sets before each critic call. Its separate-gradient helper leaves base names
disabled after the final GAN backward/commit. Those are the only two admitted
masks here; the default native TrainableWeights guard remains unchanged.

This is not a public W3 lifecycle, model loader, loss or update loop. Callers
must supply the original trainer's exact name sets and reconstruct/identify
the immutable frozen base in the outer checkpoint identity. Publication is
only after a real completed critic update with no pending gradients. Engine
counters do not prove the helper's call history: a future complete-cycle owner
must additionally check its expected update counts, data cursor and versions.
"""
from __future__ import annotations

from copy import deepcopy

from distillschedule.workloads.best_known.registry import repository_root, file_sha256
from .checkpoint import _validate_layout
from .native_state import TrainableWeights, EngineClocks, engine_state, _named


_SCHEMA = "distillschedule.original_helios_gan_critic_state.v1"
_SOURCE = {
    "reference/train_helios.py": "f1dd9c3076e737aa1e578cc47e50c14ba8fa4a7623141dd7d8258c36e4534993",
    "source/helios/utils/utils_helios_post.py": "e3ff9e5f8bf025ee740e436bc0c209c972f1e56eb0236b968fe6bd8284107004",
    "source/helios/modules/transformer_helios.py": "5d3ffe37cdea3c04a156106f6246c82f91010a1eaf0dd61529cd74083065a886",
}


def _require(condition, message):
    if not condition:
        raise ValueError("original Helios GAN critic state: " + message)


def _names(value):
    _require(not isinstance(value, str), "explicit parameter name sets required")
    try:
        names = list(value)
    except TypeError as error:
        raise ValueError("original Helios GAN critic state: explicit parameter name sets required") from error
    _require(all(isinstance(n, str) and n for n in names) and len(set(names)) == len(names),
             "invalid or duplicate parameter names")
    return sorted(names)


def _source():
    root = repository_root() / "third_party/helios_compat"
    _require(all((root / name).is_file() and file_sha256(root / name) == sha
                 for name, sha in _SOURCE.items()), "original trainer/helper/head source changed")
    return dict(_SOURCE)


class CriticPhaseWeights(TrainableWeights):
    """Reuse native tensor copying while admitting only explicitly bound phases."""

    def __init__(self, engine, *, base, extra, frozen, separate):
        super().__init__(engine)
        _require(type(separate) is bool, "is_separate_gan_grad must be an explicit bool")
        self.base, self.extra, self.frozen = _names(base), _names(extra), _names(frozen)
        _require(self.base and self.extra and not set(self.base) & set(self.extra)
                 and not set(self.frozen) & (set(self.base) | set(self.extra)),
                 "base/extra/frozen parameter names overlap or omit a trainable role")
        self.separate = separate
        self.policy = {"schema": _SCHEMA, "base": self.base, "extra": self.extra,
                       "frozen": self.frozen, "is_separate_gan_grad": separate, "source": _source()}
        model, _, _ = _named(engine)
        self.parameters = {n: id(p) for n, p in model.named_parameters()}
        self.groups = [[id(p) for p in group["params"]] for group in engine.optimizer.param_groups]
        self.state_dict()  # Fail closed at binding, including a post-update owner.

    def _phase(self, flags):
        ready = {n: n not in self.frozen for n in self.parameters}
        if flags == ready:
            return "ready"
        complete = {n: n in self.extra for n in self.parameters}
        if self.separate and flags == complete:
            return "separate_gan_complete"
        raise ValueError("original Helios GAN critic state: illegal critic trainability phase")

    def _current(self):
        model, selected, _ = _named(self.engine)
        owned = set(self.base) | set(self.extra)
        groups = [[id(p) for p in group["params"]] for group in self.engine.optimizer.param_groups]
        _require({n: id(p) for n, p in model.named_parameters()} == self.parameters
                 and set(self.parameters) == owned | set(self.frozen)
                 and set(selected) == owned and groups == self.groups
                 and len([p for group in groups for p in group]) == len(owned),
                 "base/extra/frozen names or live optimizer/Parameter ownership changed")
        _require(_source() == self.policy["source"], "bound original source changed")
        value = super().state_dict()
        value.update(policy=deepcopy(self.policy), phase=self._phase(value["requires_grad"]))
        return value

    def state_dict(self):
        return self._current()

    def validate_state_dict(self, value):
        # The fresh receiver may have zero clocks, but must itself be quiescent.
        # Check without modifying flags or hiding a live gradient/accumulation.
        EngineClocks(self.engine).validate_checkpoint_boundary()
        current = self._current()
        _require(isinstance(value, dict) and value.keys() == current.keys(), "phase state fields changed")
        _validate_layout(current["policy"], value["policy"], "original GAN phase policy")
        _require(value["policy"] == self.policy, "base/extra/frozen names, source or separate-gradient policy changed")
        _validate_layout(current["tensors"], value["tensors"], "original GAN critic trainables")
        _validate_layout(current["buffers"], value["buffers"], "original GAN critic buffers")
        _validate_layout(current["training"], value["training"], "original GAN module modes")
        _validate_layout(current["requires_grad"], value["requires_grad"], "original GAN trainability")
        _require(value["phase"] == self._phase(value["requires_grad"]), "stored critic phase/mask changed")

    def load_state_dict(self, value):
        # Common checkpoint staging validates ALL views before any live change.
        # Keep native tensor/buffer/mode copies and every Parameter/hook/grad
        # object; only the admitted original phase flags need an extra restore.
        super().load_state_dict(value)
        model, _, _ = _named(self.engine)
        for name, parameter in model.named_parameters():
            parameter.requires_grad = value["requires_grad"][name]

    def validate_checkpoint_boundary(self):
        self._current()


class CompletedCriticClocks(EngineClocks):
    """Keep the native clock schema, but do not publish/restore initial state."""

    def validate_state_dict(self, value):
        _validate_layout(self.state_dict(), value, "original GAN clocks")
        super().validate_state_dict(value)
        counters, batch = value["counters"], value["policy"]["global_batch"]
        _require(counters["global_steps"] >= counters["_step_calls"] >= 1 and batch > 0
                 and counters["global_samples"] == counters["global_steps"] * batch,
                 "checkpoint requires consistent clocks for a completed critic update, not an initial owner")

    def validate_checkpoint_boundary(self):
        super().validate_checkpoint_boundary()
        self.validate_state_dict(self.state_dict())


def critic_state(engine, *, gan_base_critic_trainable_params, gan_extra_critic_trainable_params,
                 frozen_parameter_names, is_separate_gan_grad):
    """Bind actual original name ownership, never infer it from current flags.

    Save/restore with the common atomic checkpoint API. A zero-clock, correctly
    constructed new owner is a valid receiver, not a publishable checkpoint.
    Only the critic is bound; Generator/EMA/data/whole-cycle integration is not
    supplied by this view, and no existing frozen two-cycle plan is widened.
    """
    state = engine_state({"critic": engine})
    state["critic.weights"] = CriticPhaseWeights(engine, base=gan_base_critic_trainable_params,
        extra=gan_extra_critic_trainable_params, frozen=frozen_parameter_names,
        separate=is_separate_gan_grad)
    state["critic.clocks"] = CompletedCriticClocks(engine)
    return state
