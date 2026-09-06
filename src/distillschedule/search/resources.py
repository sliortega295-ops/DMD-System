"""Search-only node telemetry. No Torch import, CUDA context or synchronization."""
from __future__ import annotations

import csv
import io
from pathlib import Path
import subprocess
import threading
import time

from distillschedule.workloads.best_known.registry import write_new_json


class ResourceObserver:
    """Two-second driver/cgroup samples spanning construction through teardown.

    Sampled minima are not continuous exact HBM extrema. Native allocator peaks
    remain separate. This identical observer is enabled for every search arm,
    not silently installed in canonical/default timing paths.
    """
    def __init__(self, path: Path, run_id: str, node_rank: int):
        self.path, self.run_id, self.node_rank = path, run_id, node_rank
        self.rows, self.errors = [], []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def sample(self) -> None:
        try:
            raw = subprocess.check_output([
                "nvidia-smi", "--query-gpu=index,uuid,memory.total,memory.free",
                "--format=csv,noheader,nounits"], text=True, timeout=10)
            gpus = [{"device": int(i), "uuid": uuid.strip(),
                     "total_bytes": int(total) * 2**20, "free_bytes": int(free) * 2**20}
                    for i, uuid, total, free in csv.reader(io.StringIO(raw))]
            if {r["device"] for r in gpus} != set(range(8)):
                raise ValueError("observer requires all eight allocated GPUs")
            root = Path("/sys/fs/cgroup")
            self.rows.append({"time_unix_ns": time.time_ns(), "gpus": gpus,
                "cgroup_current_bytes": int((root / "memory.current").read_text()),
                "cgroup_events": {k: int(v) for k, v in
                                  (line.split() for line in (root / "memory.events").read_text().splitlines())}})
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            self.errors.append(str(error))

    def _loop(self) -> None:
        while not self.stop_event.wait(2.0):
            self.sample()

    def __enter__(self):
        self.sample()
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop_event.set()
        self.thread.join(timeout=12)
        self.sample()
        write_new_json(self.path, {"schema_version": "distillschedule.search.resources.v1",
            "status": "PASS" if self.rows and not self.errors and not self.thread.is_alive() else "FAIL",
            "run_id": self.run_id, "node_rank": self.node_rank, "period_seconds": 2,
            "scope": "SAMPLED_DRIVER_AND_NODE_CGROUP_NOT_CONTINUOUS_EXTREMA",
            "errors": self.errors, "samples": self.rows})
