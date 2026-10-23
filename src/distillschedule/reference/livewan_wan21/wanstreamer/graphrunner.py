"""CUDA graph capture for steady state streaming forwards."""
import torch


class GraphedForward:
    """Captures `fn(z, tbl, e, e0) -> v` for one fixed streaming configuration."""

    def __init__(self, fn, z_shape, tbl_shape, e_shape, e0_shape, device,
                 dtype=torch.float32, warmup=3):
        self.fn = fn
        self.device = device
        self.z = torch.zeros(z_shape, device=device, dtype=dtype)
        self.tbl = torch.zeros(tbl_shape, device=device, dtype=torch.complex64)
        self.e = torch.zeros(e_shape, device=device, dtype=torch.float32)
        self.e0 = torch.zeros(e0_shape, device=device, dtype=torch.float32)
        self.graph = None
        self.warmup = warmup
        self.out = None
        self.key = None

    def capture(self):
        # allocate backend workspaces before capture
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(self.warmup):
                out = self.fn(self.z, self.tbl, self.e, self.e0)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.out = self.fn(self.z, self.tbl, self.e, self.e0)
        return self

    def __call__(self, z, tbl, e, e0):
        self.z.copy_(z)
        self.tbl.copy_(tbl)
        self.e.copy_(e)
        self.e0.copy_(e0)
        self.graph.replay()
        return self.out


class SteadyStateGraphs:
    """Cache one graph per steady state shape."""

    def __init__(self, enabled=True, max_graphs=4):
        self.enabled = enabled
        self.graphs = {}
        self.max_graphs = max_graphs
        self.captures = 0
        self.replays = 0

    def get(self, key, make):
        if not self.enabled:
            return None
        g = self.graphs.get(key)
        if g is None:
            if len(self.graphs) >= self.max_graphs:
                return None
            g = make().capture()
            self.graphs[key] = g
            self.captures += 1
        self.replays += 1
        return g
