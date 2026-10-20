"""C26 standalone exact FA2 selective-checkpoint canary, default uninstalled.

Only the native custom forward op's outputs are saved by PyTorch SAC. QKV,
projections, backward, RNG policy, and the checkpointed block stay unchanged.
Storage is local to one checkpoint, never shared across samples or updates.
"""
from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
from typing import Any


INTERFACE_SHA256 = "a49ae75310beeea12a161d42d586e3620802f457e81b775b954f97970689904b"
OP_NAME = "_flash_attn2_cuda_a511a4c::_flash_attn_forward"
PLAN_CONTRACT = {
    "enabled": True,
    "name": "critic_fa2_exact_gpu_checkpoint_outputs_v1",
    "scope": "score_rank_critic_grad_self_attention_only",
    "interface_sha256": INTERFACE_SHA256,
    "operator": OP_NAME,
    "cache_location": "cuda_per_checkpoint",
    "query_tokens": 19320,
    "key_tokens": 19320,
    "allow_cache_entry_mutation": False,
    "preserve_rng_state": True,
    "application_calls_unchanged": True,
    "cross_attention_recomputed": True,
}


def plan_contract(query_tokens: int = 19320):
    if type(query_tokens) is not int or query_tokens not in (19320, 16560):
        raise RuntimeError("unregistered ORIGINAL LiveWan attention geometry")
    return {**PLAN_CONTRACT, "query_tokens": query_tokens, "key_tokens": query_tokens}


def discover_flash_forward(flash_module: Any) -> tuple[Any, dict[str, Any]]:
    """Follow the actual public function to its registered op; reject drift."""
    function = flash_module.flash_attn_func
    cls = function.__globals__["FlashAttnFunc"]
    source = Path(inspect.getfile(cls)).resolve()
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    if sha != INTERFACE_SHA256:
        raise RuntimeError(f"FA2 interface source changed: {sha}")
    op = cls.forward.__globals__["_flash_attn_forward"]._opoverload
    if op._schema.name != OP_NAME or op._schema.is_mutable:
        raise RuntimeError(f"FA2 custom operator changed: {op._schema}")
    return op, {"interface_path": str(source), "interface_sha256": sha,
                "operator": str(op), "schema": str(op._schema)}


def attention_metadata(torch: Any, args: tuple, kwargs: dict) -> tuple:
    """Check host metadata only; no sync, tensor-value reads, copies or hashes."""
    names = ("q", "k", "v", "dropout_p", "softmax_scale", "causal",
             "window_size_left", "window_size_right", "softcap", "alibi_slopes",
             "return_softmax")
    if len(args) > len(names) or set(kwargs) - set(names):
        raise RuntimeError("FA2 argument schema changed")
    fields = dict(zip(names, args))
    if set(fields) & set(kwargs):
        raise RuntimeError("duplicate FA2 argument")
    fields.update(kwargs)
    if set(fields) != set(names):
        raise RuntimeError("FA2 argument inventory incomplete")
    q, k, v = (fields[key] for key in names[:3])
    if (any(t.ndim != 4 or not t.is_cuda or t.dtype != torch.bfloat16 for t in (q, k, v))
        or any(t.device != q.device for t in (k, v))
        or q.shape[0] not in (1, 2) or q.shape[2:] != (12, 128)
        or k.shape != v.shape or k.shape[0] != q.shape[0] or k.shape[2:] != q.shape[2:]
        or any(t.stride(-1) != 1 for t in (q, k, v))
        or fields["dropout_p"] != 0.0 or fields["causal"] is not False
        or (fields["window_size_left"], fields["window_size_right"]) != (-1, -1)
        or fields["softcap"] != 0.0 or fields["alibi_slopes"] is not None
        or fields["return_softmax"] is not False
        or fields["softmax_scale"] != 128 ** (-0.5)):
        raise RuntimeError("Critic FA2 selective-checkpoint contract changed")
    return tuple((tuple(t.shape), tuple(t.stride()), str(t.dtype), str(t.device))
                 for t in (q, k, v))


class AttentionCheckpointContexts:
    """A fresh SAC context pair per checkpoint, with a small audit ledger.

    PyTorch enforces mutation protection and ordered cached-output consumption.
    Our ledger additionally verifies operation metadata at each replay position.
    It stores no tensors. Production installation is intentionally separate.
    """
    def __init__(self, torch: Any, op: Any, identity: dict[str, Any], *, self_attention_only: bool = False,
                 query_tokens: int = 19320, metadata_reader: Any = None,
                 receipt_name: str = PLAN_CONTRACT["name"]):
        self.torch = torch
        self.op = op
        self.identity = identity
        self.self_attention_only = self_attention_only
        self.metadata_reader = attention_metadata if metadata_reader is None else metadata_reader
        self.receipt_name = receipt_name
        self.query_tokens = plan_contract(query_tokens)["query_tokens"]
        self.checkpoints: list[dict[str, Any]] = []

    def __call__(self):
        from torch.utils.checkpoint import CheckpointPolicy, create_selective_checkpoint_contexts

        ledger: dict[str, Any] = {"original": [], "replay_calls": 0, "failures": 0}
        self.checkpoints.append(ledger)

        def policy(ctx: Any, op: Any, *args: Any, **kwargs: Any):
            if op != self.op:
                return CheckpointPolicy.PREFER_RECOMPUTE
            try:
                metadata = self.metadata_reader(self.torch, args, kwargs)
                if self.self_attention_only:
                    q_shape, k_shape = metadata[0][0], metadata[1][0]
                    if q_shape[1] != self.query_tokens or k_shape[1] not in (self.query_tokens, 512):
                        raise RuntimeError("C26 unregistered Critic attention sequence shape")
                    if k_shape[1] == 512:
                        return CheckpointPolicy.PREFER_RECOMPUTE
                if ctx.is_recompute:
                    i = ledger["replay_calls"]
                    if i >= len(ledger["original"]) or metadata != ledger["original"][i]:
                        raise RuntimeError("FA2 checkpoint replay operation identity changed")
                    ledger["replay_calls"] += 1
                else:
                    ledger["original"].append(metadata)
            except Exception:
                ledger["failures"] += 1
                raise
            return CheckpointPolicy.MUST_SAVE

        return create_selective_checkpoint_contexts(policy, allow_cache_entry_mutation=False)

    def receipt(self) -> dict[str, Any]:
        return {"name": self.receipt_name,
                "source": self.identity, "checkpoint_calls": len(self.checkpoints),
                "original_fa_calls": sum(len(x["original"]) for x in self.checkpoints),
                "replay_fa_cache_reads": sum(x["replay_calls"] for x in self.checkpoints),
                "contract_failures": sum(x["failures"] for x in self.checkpoints),
                "allow_cache_entry_mutation": False, "cpu_offload": False,
                "rng_preservation": "UNCHANGED_CHECKPOINT_DEFAULT_TRUE",
                "self_attention_only": self.self_attention_only,
                "query_tokens": self.query_tokens, "key_tokens": self.query_tokens,
                "production_installed": False}


class CriticAttentionCheckpointProxy:
    """Opt-in process-local blockcausal.ckpt proxy; never patch global Torch."""
    def __init__(self, checkpoint_module: Any, torch: Any, flash_module: Any, *, query_tokens: int = 19320):
        self.module = checkpoint_module
        op, identity = discover_flash_forward(flash_module)
        self.contexts = AttentionCheckpointContexts(torch, op, identity, self_attention_only=True, query_tokens=query_tokens)
        self.receipt_errors = runtime_receipt_errors

    def checkpoint(self, function: Any, *args: Any, **kwargs: Any):
        if (kwargs.get("use_reentrant") is not False or "context_fn" in kwargs
            or "preserve_rng_state" in kwargs):
            raise RuntimeError("C26 checkpoint call site/RNG/context changed")
        return self.module.checkpoint(function, *args, context_fn=self.contexts, **kwargs)

    def receipt(self, *, expected_calls: int):
        r = self.contexts.receipt()
        r.update(enabled=True, execution_status="PASS", production_installed=True,
                 expected_checkpoint_calls=expected_calls)
        if self.receipt_errors(r, rank=8, expected_calls=expected_calls, query_tokens=self.contexts.query_tokens):
            raise RuntimeError(f"C26 checkpoint replay census failed: {r}")
        return r


def runtime_receipt_errors(r: dict[str, Any], *, rank: int, expected_calls: int, query_tokens: int = 19320) -> list[str]:
    if rank < 8:
        return [] if r.get("execution_status") == "NOT_APPLICABLE" and r.get("checkpoint_calls") == 0 else ["Generator SAC changed"]
    if (expected_calls <= 0 or r.get("name") != PLAN_CONTRACT["name"]
        or (query_tokens != 19320 and (r.get("query_tokens") != query_tokens or r.get("key_tokens") != query_tokens))
        or r.get("source", {}).get("interface_sha256") != INTERFACE_SHA256
        or r.get("execution_status") != "PASS" or r.get("production_installed") is not True
        or r.get("self_attention_only") is not True or r.get("contract_failures") != 0
        or r.get("checkpoint_calls") != expected_calls
        or r.get("original_fa_calls") != expected_calls
        or r.get("replay_fa_cache_reads") != expected_calls
        or r.get("expected_checkpoint_calls") != expected_calls
        or r.get("allow_cache_entry_mutation") is not False
        or r.get("cpu_offload") is not False):
        return ["C26 selective-checkpoint source/replay/census contract failed"]
    return []
