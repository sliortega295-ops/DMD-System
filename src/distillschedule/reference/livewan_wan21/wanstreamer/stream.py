"""Few step block causal streaming used by inference and self forcing training.

    x0 = z - t * G(z, t | clean K/V)
    z_next = (1 - t_next) * x0 + t_next * noise

Finished blocks are rerun at `t=0` before their K/V is committed. The world K/V
stays pinned while a bounded event window keeps memory use constant.
"""
import time

import torch

from . import blockcausal as bc
from .core import make_rope_table, make_cache, latent_geometry
from .graphrunner import GraphedForward, SteadyStateGraphs


def few_step_schedule(num_steps, t_max=1.0, shift=1.0):
    """Return descending flow fractions for the requested sampling schedule."""
    ts = [t_max * (num_steps - i) / num_steps for i in range(num_steps)]
    if shift == 1.0:
        return ts
    return [shift * t / (1 + (shift - 1) * t) for t in ts]


class FewStepStreamer:
    def __init__(self, model, width=640, height=368, max_frames=256,
                 device='cuda', dtype=torch.bfloat16, window_frames=None,
                 time_scale=1000.0, cache_frames=None, block_frames=3,
                 num_steps=4, rope=None, shift=1.0, sampler='renoise',
                 cuda_graphs=False, noisy_context=False):
        self.model = model
        self.device = device
        self.dtype = dtype
        self.time_scale = time_scale
        self.block_frames = block_frames
        self.schedule = few_step_schedule(num_steps, shift=shift)
        self.sampler = sampler
        self.h_lat, self.w_lat, self.hp, self.wp = latent_geometry(width, height)
        self.S = self.hp * self.wp
        self.max_frames = max_frames
        self.window_frames = window_frames
        self.rope = rope if rope is not None else make_rope_table(
            model, self.hp, self.wp, max_frames, device)
        self.cache = make_cache(model, self.S, cache_frames or max_frames,
                                device, dtype)
        self.kv = bc.BufferKV(self.cache)
        self.n_world = 0
        self.n_frames = 0
        # training uses this gap to simulate evicted events
        self.rope_gap = 0
        # per channel moment matching toward the world
        self.latent_norm = 0.0
        self.ref_stats = None
        self.graphs = SteadyStateGraphs() if cuda_graphs else None
        # noisy context skips the clean K/V pass and changes training semantics
        self.noisy_context = noisy_context

    def set_text(self, ctx_emb, ctx_lens=None):
        self.ctx = ctx_emb
        # resolve padding once to avoid a host sync during graph capture
        full = ctx_emb.shape[1]
        if ctx_lens is not None and int(torch.as_tensor(ctx_lens).min()) < full:
            self.ctx_lens = ctx_lens
        else:
            self.ctx_lens = None

    def _eager(self, z, tbl, e, e0):
        return bc.block_forward(self.model, z, 0.0, 0, self.rope, self.ctx,
                                self.ctx_lens, kv=self.kv, dtype=self.dtype,
                                time_scale=self.time_scale, tbl=tbl, emb=(e, e0))

    def _fwd(self, z, t, t_start):
        tbl = self.rope.span(t_start, z.shape[2])
        e, e0 = bc.time_embed(self.model, t, self.device, self.time_scale)
        if self.graphs is not None and not torch.is_grad_enabled():
            # cache length determines the graph shape
            g = self.graphs.get(
                (self.cache.length, z.shape[2]),
                lambda: GraphedForward(self._eager, tuple(z.shape),
                                       tuple(tbl.shape), tuple(e.shape),
                                       tuple(e0.shape), self.device))
            if g is not None:
                return g(z, tbl, e, e0)
        return self._eager(z, tbl, e, e0)

    def _commit(self, n):
        self.cache.commit(n * self.S)
        self.n_frames += n
        if self.window_frames is not None:
            self.trim_to_window(self.window_frames)

    def trim_to_window(self, window_frames):
        """Evict the oldest events so only `window_frames` of them remain.

        Training calls this only between iterations because eviction invalidates
        recorded prefixes. The world prefix is always protected.
        """
        protect = self.n_world * self.S
        budget = (self.n_world + window_frames) * self.S
        excess = self.cache.num_tokens - budget
        if excess > 0:
            self.cache.evict_front(excess, protect=protect)

    # world state
    @torch.no_grad()
    def set_world(self, world_latents):
        """Prime the cache from clean world latents, attended bidirectionally."""
        self.cache.reset()
        self.n_frames = 0
        z = world_latents.to(self.device).unsqueeze(0) \
            if world_latents.dim() == 4 else world_latents.to(self.device)
        F = z.shape[2]
        self._fwd(z, 0.0, 0)
        self._commit(F)
        self.n_world = F
        # per channel reference moments over F H W
        w = z[0].float()
        self.ref_stats = (w.mean(dim=(1, 2, 3), keepdim=True).unsqueeze(0),
                          w.std(dim=(1, 2, 3), keepdim=True).unsqueeze(0))
        return F

    def _renorm(self, z, strength):
        """Move block channel moments toward the opening world.

        This occurs before clean K/V is committed and corrects scale and offset,
        not semantic direction.
        """
        if strength <= 0 or getattr(self, 'ref_stats', None) is None:
            return z
        mu, sd = self.ref_stats
        zm = z.mean(dim=(2, 3, 4), keepdim=True)
        zs = z.std(dim=(2, 3, 4), keepdim=True)
        z_n = (z - zm) / (zs + 1e-5) * sd.to(z.dtype) + mu.to(z.dtype)
        return (1.0 - strength) * z + strength * z_n

    # event generation
    def generate_block(self, generator=None, record=None, n_frames=None):
        """Emit one streaming unit. Returns clean latents [1, C, b, H, W]."""
        b = n_frames or self.block_frames
        t0 = bc.event_rope_index(self.n_frames, self.n_world, self.rope_gap)
        z = torch.randn(1, 16, b, self.h_lat, self.w_lat, device=self.device,
                        dtype=torch.float32, generator=generator)
        sched = self.schedule
        for i, t in enumerate(sched):
            if record is not None:
                record.append({'block_start': t0, 'step': i, 't': t,
                               'z': z.detach()})
            v = self._fwd(z, t, t0).float()
            x0 = z - t * v
            tn = sched[i + 1] if i + 1 < len(sched) else 0.0
            if tn == 0.0:
                z = x0
            elif self.sampler == 'renoise':
                # renoise x0 to the next level with fresh noise
                eps = torch.randn(z.shape, device=z.device, dtype=z.dtype,
                                  generator=generator)
                z = (1 - tn) * x0 + tn * eps
            else:
                z = z + (tn - t) * v          # deterministic Euler, for the
        z = self._renorm(z, self.latent_norm)
        if not self.noisy_context:
            self._fwd(z, 0.0, t0)      # clean K/V for the committed block
        # noisy context commits the final denoising pass from the scratch region
        self._commit(b)
        return z

    @torch.no_grad()
    def stream(self, num_blocks, seed=0, log_every=0, n_frames=None):
        g = torch.Generator(device=self.device).manual_seed(seed)
        lats, times = [], []
        for u in range(num_blocks):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            z = self.generate_block(generator=g, n_frames=n_frames)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            lats.append(z[0])
            times.append(dt)
            if log_every and (u + 1) % log_every == 0:
                b = n_frames or self.block_frames
                print(f'  unit {u+1}/{num_blocks}: {dt*1000:7.1f} ms '
                      f'({dt/b*1000:6.1f} ms/latent frame, '
                      f'cache {self.cache.num_tokens} tok)', flush=True)
        return torch.cat(lats, dim=1), times

    # self forcing rollout
    @torch.no_grad()
    def rollout_record(self, num_blocks, generator=None):
        """Roll out `num_blocks` units and keep what the DMD step needs.

        Records contain each input latent, timestep and cache prefix length.
        """
        out, recs = [], []
        for _ in range(num_blocks):
            rec = []
            prefix = self.cache.num_tokens
            ev = self.cache.evictions
            z = self.generate_block(generator=generator, record=rec)
            for r in rec:
                r['prefix'] = prefix
                r['evictions'] = ev
            out.append(z)
            recs.append(rec)
        return out, recs

    def prefix_intact(self, rec_step):
        """Is this recorded step's prefix still literally `buffer[:prefix]`?

        Any later eviction makes the recorded prefix invalid for gradients.
        """
        return rec_step.get('evictions', 0) == self.cache.evictions

    def recompute_step(self, rec_step, prefix_kv):
        """Differentiably redo one recorded denoising step against a detached
        prefix. Returns that step's x0 prediction."""
        z, t = rec_step['z'], rec_step['t']
        v = bc.block_forward(self.model, z, t, rec_step['block_start'],
                             self.rope, self.ctx, self.ctx_lens, kv=prefix_kv,
                             dtype=self.dtype, time_scale=self.time_scale,
                             prefix_upto=rec_step['prefix'],
                             grad_checkpoint=True)
        return z - t * v.float()
