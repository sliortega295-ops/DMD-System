"""Opt-in host-only highwater accounting across existing native counter resets."""


class CudaHighwater:
    def __init__(self, cuda, device):
        self.cuda, self.device = cuda, device
        self.original = cuda.reset_peak_memory_stats
        self.allocated = self.reserved = self.resets = 0
        self.active = False

    def sample(self):
        self.allocated = max(self.allocated, self.cuda.max_memory_allocated(self.device))
        self.reserved = max(self.reserved, self.cuda.max_memory_reserved(self.device))

    def start(self):
        if self.active:
            raise RuntimeError('highwater recorder already active')
        self.sample()
        def reset(device=None):
            # Each worker has one assigned/current device. Unrelated resets are
            # rejected rather than silently attributed to the current GPU.
            if device is not None and str(device) not in (str(self.device), str(self.device.index)):
                raise RuntimeError('unexpected device in highwater recorder')
            self.sample()
            self.resets += 1
            return self.original(device)
        self.reset = reset
        self.cuda.reset_peak_memory_stats = reset
        self.active = True

    def close(self):
        if self.active:
            try:
                self.sample()
            finally:
                self.cuda.reset_peak_memory_stats = self.original
                self.active = False
        return dict(peak_allocated_bytes=self.allocated, peak_reserved_bytes=self.reserved,
                    native_resets_observed=self.resets,
                    scope='PYTORCH_HIGHWATER_ACROSS_OBSERVED_NATIVE_PYTHON_RESETS_INCLUDING_SETUP_AND_BOTH_CYCLES')
