from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
dist = pytest.importorskip("torch.distributed")
mp = pytest.importorskip("torch.multiprocessing")

from runtime.critic_pool import ChecksumKind
from runtime.critic_pool_transport import tensor_identity
from distillschedule.runtime.gan_pipeline_transport import LATENT_SHAPE
from distillschedule.runtime.gan_rolephase_transport import (
    RolePhaseCriticP2PTransport,
    make_phase_b_header,
    make_phase_b_route,
)


def _gloo_fifo_worker(rank: int, store_path: str) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{store_path}",
        rank=rank,
        world_size=2,
    )
    try:
        transport = RolePhaseCriticP2PTransport(
            group=dist.group.WORLD,
            device=torch.device("cpu"),
            global_rank=rank,
        )
        trace0 = bytes.fromhex("11" * 32)
        trace1 = bytes.fromhex("22" * 32)
        route0 = make_phase_b_route(
            logical_step=1000,
            logical_slot=0,
            generator_rank=0,
            critic_rank=1,
            generator_version=201,
            critic_before=1000,
            trace_sha256=trace0,
        )
        route1 = make_phase_b_route(
            logical_step=1001,
            logical_slot=0,
            generator_rank=0,
            critic_rank=1,
            generator_version=201,
            critic_before=1001,
            trace_sha256=trace1,
        )
        if rank == 0:
            pending = []
            for route, fill in ((route0, 1.0), (route1, 2.0)):
                payload = torch.full(LATENT_SHAPE, fill, dtype=torch.bfloat16)
                checksum = tensor_identity(payload, ChecksumKind.GPU_FINGERPRINT_V1)
                header = make_phase_b_header(
                    run_nonce=1234,
                    route=route,
                    payload_checksum=checksum,
                )
                pending.append(transport.send(header, payload, route=route))
            for item in pending:
                transport.reap_send_completion(item)
        else:
            first = transport.post_receive(route=route0)
            second = transport.post_receive(route=route1)
            # Multiple posted receives on one edge are legal, but consumption
            # must follow the explicit route FIFO rather than Python list order.
            try:
                transport.finish_receive(second)
            except RuntimeError as error:
                if "FIFO" not in str(error):
                    raise
            else:
                raise AssertionError("out-of-order receive consumption unexpectedly passed")
            header0, payload0 = transport.finish_receive(first)
            header1, payload1 = transport.finish_receive(second)
            route0.validate_header(header0)
            route1.validate_header(header1)
            if float(payload0.flatten()[0]) != 1.0:
                raise AssertionError("first FIFO payload identity changed")
            if float(payload1.flatten()[0]) != 2.0:
                raise AssertionError("second FIFO payload identity changed")
        transport.flush()
        transport.assert_flushed()
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_two_rank_gloo_transport_is_fifo_and_identity_bound(tmp_path: Path) -> None:
    if not dist.is_gloo_available():
        pytest.skip("Gloo backend is unavailable")
    store = tmp_path / "rolephase-gloo-store"
    mp.spawn(_gloo_fifo_worker, args=(str(store),), nprocs=2, join=True)
