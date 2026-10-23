"""One atomic native G1/C5 macrocycle, not a second training framework.

The common Program owns invocation, data, checkpoint and export. The source's
actual ``Trainer.fwdbwd_one_step`` owns conditioning, rollout, losses, backward
and clipping. This owner only preserves native Adam/EMA/iteration ordering.
"""
from contextlib import contextmanager
from copy import deepcopy
import hashlib

from .checkpoint import cpu_tree, rng_state, restore_rng, validate_rng, _validate_layout, _validate_optimizer_layout
from .self_forcing_source import original, require
from .single_device import device_scope as _device_scope

SCHEMA = 'distillschedule.official_self_forcing_owner.v1'
CUDA_TRIAL_LIMIT = ('CUDA owner trials require a capacity-bounded snapshot/restore including frozen '
                    'Teacher/VAE, native CPU EMA, optimizer and CPU staging; auto/plan/owner_execution are not admitted')


def stream_counts(args):
    return {'generator': 1, 'critic': args.dfake_gen_update_ratio}


def _hash(value):
    import torch
    return hashlib.sha256(value.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy()).hexdigest()


def plain_tensors(model):
    """Native non-buffer RoPE/scheduler/VAE tensors have explicit ownership."""
    result = {'denoising_step_list': model.denoising_step_list,
              'vae.mean': model.vae.mean, 'vae.std': model.vae.std}
    for role in ('generator', 'real_score', 'fake_score'):
        wrapper = getattr(model, role)
        result[role + '.model.freqs'] = wrapper.model.freqs
        for name in ('timesteps', 'sigmas', 'linear_timesteps_weights'):
            result[role + '.scheduler.' + name] = getattr(wrapper.scheduler, name)
    return result


class SelfForcingOwner:
    def __init__(self, model, args, source, negative_embedding, *, identity):
        import torch
        from types import SimpleNamespace
        from distillschedule.data.self_forcing_inputs import validate_row
        self.model, self.args, self.source = model, args, source
        self.device, self.dtype = model.device, model.dtype
        self.identity = deepcopy(identity)
        validate_row({'prompt_embeds': negative_embedding},
            shape=(1, model.generator.model.text_len, model.generator.model.text_dim), device=self.device)
        self.negative_embedding = negative_embedding.detach().clone()
        self.negative_hash = _hash(self.negative_embedding)
        # The original fwdbwd method calls a text encoder on a list. At this
        # explicit preencoded boundary each item is an actual native embedding;
        # the identity adapter does no tokenization or invented conditioning.
        self.config = SimpleNamespace(**{**vars(args), 'negative_prompt': self.negative_embedding[0]})
        self.native_step = self.generator_updates = self.critic_updates = self.ema_updates = 0
        self.ratio = args.dfake_gen_update_ratio
        self.max_grad_norm_generator = args.max_grad_norm_generator
        self.max_grad_norm_critic = args.max_grad_norm_critic
        self.unconditional_dict = None
        self.optimizers = {
            'generator': torch.optim.AdamW([p for p in model.generator.parameters() if p.requires_grad],
                lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay),
            'critic': torch.optim.AdamW([p for p in model.fake_score.parameters() if p.requires_grad],
                lr=args.lr_critic, betas=(args.beta1_critic, args.beta2_critic), weight_decay=args.weight_decay)}
        names = {id(p): name for name, p in model.named_parameters()}
        self.ownership = {role: [[names[id(p)] for p in group['params']] for group in optimizer.param_groups]
                          for role, optimizer in self.optimizers.items()}
        self.optimizer_activity = {role: {name: 0 for group in groups for name in group}
                                   for role, groups in self.ownership.items()}
        self.masks = {name: p.requires_grad for name, p in model.named_parameters()}
        self.frozen_hashes = {name: _hash(p) for name, p in model.state_dict().items()
                              if not self.masks.get(name, False)}
        self.plain_hashes = {name: _hash(tensor) for name, tensor in plain_tensors(model).items()}
        self.plain_specs = {name: (tensor.shape, tensor.dtype) for name, tensor in plain_tensors(model).items()}
        self.generator_ema = None
        self.ema_initialized_step = None
        if args.ema_start_step == 0 and args.ema_weight > 0:
            self._initialize_ema()
        self._rng = cpu_tree(rng_state(device=self.device))
        self.observer = None
        self._validate_live()

    @property
    def cycle(self):
        return self.native_step // self.ratio

    @property
    def step(self):
        return self.native_step

    def __deepcopy__(self, memo):
        # Generic paired trials clone whole owners. Reject before allocating a
        # second CUDA model until a capacity-bounded snapshot ABI is available.
        require(self.device.type != 'cuda', CUDA_TRIAL_LIMIT)
        result = type(self).__new__(type(self))
        memo[id(self)] = result
        result.__dict__.update(deepcopy(self.__dict__, memo))
        return result

    def admit_execution(self, mode, implementation):
        require(self.device.type != 'cuda' or (mode in ('reference', 'manual')
                and 'owner_execution' not in implementation), CUDA_TRIAL_LIMIT)

    def _event(self, name, value):
        if self.observer is not None:
            self.observer(name, value)

    @contextmanager
    def random_scope(self):
        with _device_scope(self.device):
            outer = rng_state(device=self.device)
            restore_rng(self._rng)
            try:
                yield
            finally:
                self._rng = cpu_tree(rng_state(device=self.device))
                restore_rng(outer)

    def _ema_container(self):
        import torch
        # FSDP.summon_full_params is a no-op on this unsharded nn.Module tree.
        # This lets the original EMA_FSDP arithmetic and parameter naming run
        # verbatim; the original shadow is CPU FP32 even for a CUDA generator.
        container = torch.nn.Module()
        container.add_module('module', self.model.generator)
        return container

    def _initialize_ema(self):
        self.generator_ema = original(self.source.root).ema.EMA_FSDP(
            self._ema_container(), decay=self.args.ema_weight)
        self.ema_initialized_step = self.native_step

    def runtime_tensors(self):
        """Compute tensors only; full host RNG/Adam-step identity still saves."""
        yield from self.model.parameters()
        yield from self.model.buffers()
        yield from plain_tensors(self.model).values()
        yield self.negative_embedding
        if self.unconditional_dict is not None:
            yield from self.unconditional_dict.values()
        for p in self.model.parameters():
            if p.grad is not None:
                yield p.grad
        for optimizer in self.optimizers.values():
            for state in optimizer.state.values():
                for name, value in state.items():
                    if name != 'step':
                        yield value
        # Native EMA_FSDP intentionally owns CPU FP32 shadows. They, RNG and
        # Adam's noncapturable scalar step are host state, not compute residency.

    def _validate_live(self):
        import torch
        from .self_forcing import _admit_device, runtime_identity
        require(_admit_device({'device': str(self.device)}) == self.model.device
                and self.identity['placement'] == dict(device=str(self.device), layout='SINGLE_DEVICE_RESIDENT',
                                                       distributed=False, amp=False)
                and self.model.args is self.args and self.dtype == self.model.dtype == torch.float32,
                'live native model/device/FP32 placement changed')
        require(self.identity['runtime'] == runtime_identity(), 'live numerical runtime/precision policy changed')
        self.source.validate()
        require(type(self.native_step) is int and self.native_step % self.ratio == 0
                and 0 <= self.native_step <= self.args.train_iters
                and self.generator_updates == self.cycle and self.critic_updates == self.native_step,
                'owner is not at a complete native macrocycle')
        require(self.model.inference_pipeline is None, 'checkpoint requires discarded ephemeral rollout caches')
        require(vars(self.args) == self.identity['settings'], 'native settings mutated outside the owned lifecycle')
        require(all(p.grad is None for p in self.model.parameters()), 'checkpoint requires cleared gradients')
        require(self.masks == {name: p.requires_grad for name, p in self.model.named_parameters()},
                'native role parameter masks changed')
        require(all(type(t) in (torch.Tensor, torch.nn.Parameter) and t.device == self.device
                    and t.layout == torch.strided for t in self.runtime_tensors()),
                'plain dense single-device placement required; no distributed, offload or AMP layout')
        require(all(not t.is_floating_point() or t.dtype == torch.float32
                    for t in (*self.model.parameters(), *self.model.buffers()))
                and self.negative_embedding.dtype == torch.float32,
                'native module/condition FP32 dtype changed')
        native_plain = plain_tensors(self.model)
        require(self.plain_specs == {name: (tensor.shape, tensor.dtype) for name, tensor in native_plain.items()},
                'native non-buffer shape/dtype changed')
        require(self.model.scheduler is self.model.generator.scheduler, 'native generator scheduler alias changed')
        require(all(state.get('step') is None or (type(state['step']) is torch.Tensor
                    and state['step'].device.type == 'cpu' and state['step'].dtype == torch.float32
                    and state['step'].ndim == 0) for optimizer in self.optimizers.values()
                    for state in optimizer.state.values()), 'AdamW noncapturable step metadata must remain FP32 on CPU')
        names = {id(p): name for name, p in self.model.named_parameters()}
        require(self.ownership == {role: [[names.get(id(p)) for p in group['params']]
                    for group in optimizer.param_groups] for role, optimizer in self.optimizers.items()},
                'optimizer role/name ownership changed')
        for role, optimizer in self.optimizers.items():
            _validate_optimizer_layout(optimizer, optimizer.state_dict(), 'Self-Forcing live ' + role)
        require(all(name.startswith('generator.') for group in self.ownership['generator'] for name in group)
                and all(name.startswith('fake_score.') for group in self.ownership['critic'] for name in group),
                'teacher or codec entered a trainable optimizer')
        model_state = self.model.state_dict()
        require(all(_hash(model_state[name]) == digest for name, digest in self.frozen_hashes.items())
                and all(_hash(native_plain[name]) == digest for name, digest in self.plain_hashes.items()),
                'live frozen teacher/VAE or native non-buffer tensor changed')
        require(self.generator_ema is None or self.generator_ema.decay == self.args.ema_weight,
                'native EMA policy changed')
        if self.generator_ema is not None:
            parameters = dict(self.model.generator.named_parameters())
            require(self.generator_ema.shadow.keys() == parameters.keys()
                    and all(type(t) is torch.Tensor and t.device.type == 'cpu' and t.dtype == torch.float32
                            and t.shape == parameters[name].shape and not t.requires_grad
                            for name, t in self.generator_ema.shadow.items()),
                    'native EMA_FSDP shadow must remain detached CPU FP32 host state')
        from distillschedule.data.self_forcing_inputs import validate_row
        validate_row({'prompt_embeds': self.negative_embedding},
            shape=(1, self.model.generator.model.text_len, self.model.generator.model.text_dim), device=self.device)
        require(_hash(self.negative_embedding) == self.negative_hash, 'native encoded negative conditioning changed')
        require(vars(self.config).keys() == vars(self.args).keys()
                and all(value == getattr(self.args, name) for name, value in vars(self.config).items()
                        if name != 'negative_prompt'), 'native trainer settings changed')
        negative = self.config.negative_prompt
        require(type(negative) is torch.Tensor and negative.device == self.device and negative.dtype == torch.float32
                and torch.equal(negative, self.negative_embedding[0]), 'native negative conditioning boundary changed')
        if self.unconditional_dict is not None:
            validate_row(self.unconditional_dict, shape=(self.args.batch_size,
                self.model.generator.model.text_len, self.model.generator.model.text_dim), device=self.device)
            require(torch.equal(self.unconditional_dict['prompt_embeds'],
                                self.negative_embedding.repeat(self.args.batch_size, 1, 1)),
                    'native unconditional cache differs from exact encoded negative prompt')

    def validate_feeds(self, feeds):
        from distillschedule.data.self_forcing_inputs import validate_row
        counts = stream_counts(self.args)
        require(isinstance(feeds, dict) and set(feeds) == set(counts), 'independent G/C input streams required')
        for role, count in counts.items():
            require(isinstance(feeds[role], list) and len(feeds[role]) == count,
                    'native prompt consumption changed for ' + role)
            for row in feeds[role]:
                validate_row(row, shape=(self.args.batch_size, self.model.generator.model.text_len,
                                         self.model.generator.model.text_dim), device=self.device)

    def _forward_backward(self, row, generator):
        native = original(self.source.root).trainer.Trainer.fwdbwd_one_step
        return native(self, {'prompts': list(row['prompt_embeds'].unbind(0))}, generator)

    def _commit(self, role):
        optimizer = self.optimizers[role]
        for group, names in zip(optimizer.param_groups, self.ownership[role], strict=True):
            for p, name in zip(group['params'], names, strict=True):
                self.optimizer_activity[role][name] += int(p.grad is not None)
        optimizer.step()
        if role == 'generator':
            self.generator_updates += 1
            if self.generator_ema is not None:
                self.generator_ema.update(self._ema_container())
                self.ema_updates += 1
        else:
            self.critic_updates += 1

    def run(self, feeds, *, cycle):
        self._validate_live()
        require(type(cycle) is int and cycle == self.cycle, 'native macrocycle reordered')
        require(self.native_step + self.ratio <= self.args.train_iters, 'native iteration budget exhausted')
        self.validate_feeds(feeds)
        rows = []
        with self.random_scope():
            for offset in range(self.ratio):
                generator_turn = self.native_step % self.ratio == 0
                losses, norms = {}, {}
                if generator_turn:
                    self.optimizers['generator'].zero_grad(set_to_none=True)
                    result = self._forward_backward(feeds['generator'][0], True)
                    self._event('generator_gradients', {name: p.grad for name, p in self.model.generator.named_parameters()})
                    self._event('generator_rollout_cache', self._cache_snapshot())
                    losses['generator_loss'] = float(result['generator_loss'].detach())
                    norms['generator_grad_norm'] = float(result['generator_grad_norm'])
                    self._commit('generator')
                    self._event('generator_committed', self.model.generator.state_dict())
                self.optimizers['critic'].zero_grad(set_to_none=True)
                # A new row AND a new native self-rollout: no G sample reuse.
                result = self._forward_backward(feeds['critic'][offset], False)
                self._event('critic_gradients', {name: p.grad for name, p in self.model.fake_score.named_parameters()})
                self._event('critic_rollout_cache', self._cache_snapshot())
                losses['critic_loss'] = float(result['critic_loss'].detach())
                norms['critic_grad_norm'] = float(result['critic_grad_norm'])
                self._commit('critic')
                row = dict(native_step=self.native_step, generator_turn=generator_turn, losses=losses, **norms)
                self.native_step += 1
                # Original train increments the step BEFORE EMA initialization.
                if self.native_step >= self.args.ema_start_step and self.generator_ema is None and self.args.ema_weight > 0:
                    self._initialize_ema()
                rows.append(row)
                self._event('iteration_complete', row)
        # Cache allocation is recreated by every native training rollout. No
        # ephemeral cache or outstanding autograd graph crosses Program calls.
        self.model.inference_pipeline = None
        for optimizer in self.optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        self._validate_live()
        return dict(iterations=rows, native_step=self.native_step, generator_updates=self.generator_updates,
            critic_updates=self.critic_updates, ema_updates=self.ema_updates,
            consumed={role: self.cycle * count for role, count in stream_counts(self.args).items()})

    def _cache_snapshot(self):
        pipeline = self.model.inference_pipeline
        return {'kv': pipeline.kv_cache1, 'crossattn': pipeline.crossattn_cache}

    def state_dict(self):
        self._validate_live()
        return dict(schema_version=SCHEMA, identity=deepcopy(self.identity), native_step=self.native_step,
            generator_updates=self.generator_updates, critic_updates=self.critic_updates,
            ema_updates=self.ema_updates, ema_initialized_step=self.ema_initialized_step,
            ema=None if self.generator_ema is None else self.generator_ema.state_dict(),
            consumed={role: self.cycle * count for role, count in stream_counts(self.args).items()},
            model=self.model.state_dict(), plain=plain_tensors(self.model),
            training={name: module.training for name, module in self.model.named_modules()},
            requires_grad={name: p.requires_grad for name, p in self.model.named_parameters()},
            optimizer_ownership=deepcopy(self.ownership), optimizer_activity=deepcopy(self.optimizer_activity),
            optimizers={name: optimizer.state_dict() for name, optimizer in self.optimizers.items()},
            negative_embedding=self.negative_embedding, unconditional_dict=self.unconditional_dict, rng=cpu_tree(self._rng))

    def validate_state_dict(self, value):
        import torch
        from torch.utils._pytree import tree_leaves
        current = self.state_dict()
        require(isinstance(value, dict) and value.keys() == current.keys()
                and value['schema_version'] == SCHEMA and value['identity'] == self.identity,
                'checkpoint source/model/runtime/settings identity changed')
        # A checkpoint may stage compute tensors on CPU, but never contain a
        # meta/sparse/distributed or unrelated-device tensor that fails midway
        # through a load. Validate the entire source tree before any mutation.
        require(all(type(t) is torch.Tensor and t.layout == torch.strided and not t.requires_grad
                    and (t.device.type == 'cpu' or t.device == self.device)
                    for t in tree_leaves(value) if isinstance(t, torch.Tensor)),
                'checkpoint requires dense CPU or owned-device staging tensors')
        step = value['native_step']
        require(type(step) is int and 0 <= step <= self.args.train_iters and step % self.ratio == 0,
                'checkpoint is not a complete native macrocycle')
        cycle = step // self.ratio
        require(type(value['generator_updates']) is int and type(value['critic_updates']) is int
                and value['generator_updates'] == cycle and value['critic_updates'] == step
                and value['consumed'] == {role: cycle * count for role, count in stream_counts(self.args).items()},
                'native optimizer/data clocks changed')
        require(all(type(count) is int for count in value['consumed'].values()), 'native data clocks must be integer counters')
        require(value['requires_grad'] == self.masks and value['training'] == current['training'],
                'native teacher/fake/generator modes or masks changed')
        require(all(type(flag) is bool for flag in [*value['requires_grad'].values(), *value['training'].values()]),
                'native modes and masks must be boolean')
        _validate_layout(current['model'], value['model'], 'Self-Forcing native model')
        _validate_layout(current['plain'], value['plain'], 'Self-Forcing native non-buffer tensors')
        require(all(_hash(value['model'][name]) == digest for name, digest in self.frozen_hashes.items())
                and all(_hash(value['plain'][name]) == digest for name, digest in self.plain_hashes.items()),
                'frozen teacher/VAE/RoPE/scheduler tensor changed')
        _validate_layout(self.negative_embedding, value['negative_embedding'], 'negative conditioning')
        negative_cpu = self.negative_embedding.detach().cpu()
        require(torch.equal(negative_cpu, value['negative_embedding'].cpu()), 'negative conditioning changed')
        expected_condition = {'prompt_embeds': negative_cpu.repeat(self.args.batch_size, 1, 1)} if step else None
        _validate_layout(expected_condition, value['unconditional_dict'], 'native unconditional cache')
        require(expected_condition is None or torch.equal(expected_condition['prompt_embeds'],
                                                        value['unconditional_dict']['prompt_embeds'].cpu()),
                'native unconditional cache differs from exact encoded negative prompt')
        ema_live = self.args.ema_weight > 0 and step >= self.args.ema_start_step
        initialized = self.args.ema_start_step if ema_live else None
        expected_updates = sum(1 for index in range(0, step, self.ratio)
                               if self.args.ema_weight > 0 and index >= self.args.ema_start_step)
        require(value['ema_initialized_step'] == initialized and type(value['ema_updates']) is int
                and value['ema_updates'] == expected_updates, 'EMA initialization/update clocks changed')
        require(value['ema_initialized_step'] is None or type(value['ema_initialized_step']) is int,
                'EMA initialization step must be an integer')
        ema_template = {name: p.detach().float().cpu() for name, p in self.model.generator.named_parameters()} if ema_live else None
        _validate_layout(ema_template, value['ema'], 'native EMA shadow')
        require(value['ema'] is None or all(t.device.type == 'cpu' and not t.requires_grad for t in value['ema'].values()),
                'checkpoint native EMA shadow must remain CPU host state')
        require(value['optimizer_ownership'] == self.ownership
                and set(value['optimizers']) == set(self.optimizers)
                and set(value['optimizer_activity']) == set(self.optimizers), 'optimizer ownership changed')
        for role, optimizer in self.optimizers.items():
            saved = value['optimizers'][role]
            expected_groups = current['optimizers'][role]['param_groups']
            require(isinstance(saved, dict) and set(saved) == {'state', 'param_groups'}
                    and len(saved['param_groups']) == len(expected_groups), 'invalid native AdamW groups')
            for group, expected in zip(saved['param_groups'], expected_groups, strict=True):
                require(group == expected, 'native constant-LR AdamW policy/parameter layout changed')
            activity = value['optimizer_activity'][role]
            maximum = cycle if role == 'generator' else step
            require(isinstance(activity, dict) and activity.keys() == self.optimizer_activity[role].keys()
                    and all(type(count) is int and 0 <= count <= maximum for count in activity.values()),
                    'native per-parameter gradient participation changed')
            ids = {index: name for group, names in zip(expected_groups, self.ownership[role], strict=True)
                   for index, name in zip(group['params'], names, strict=True)}
            require(set(saved['state']) == {index for index, name in ids.items() if activity[name]},
                    'native AdamW slots missing or attached to unused parameters')
            for index, state in saved['state'].items():
                require(set(state) == {'step', 'exp_avg', 'exp_avg_sq'}
                        and type(state['step']) is torch.Tensor and state['step'].dtype == torch.float32
                        and state['step'].device.type == 'cpu' and state['step'].ndim == 0
                        and float(state['step']) == activity[ids[index]],
                        'native AdamW parameter update clock changed')
            _validate_optimizer_layout(optimizer, saved, 'Self-Forcing ' + role)
        require(all(bool(torch.isfinite(tensor).all()) for tensor in tree_leaves(value)
                    if isinstance(tensor, torch.Tensor) and (tensor.is_floating_point() or tensor.is_complex())),
                'nonfinite checkpoint tensor')
        try:
            require(set(value['rng']) == set(self._rng), 'native RNG fields changed')
            require(('cuda' in value['rng']) == (self.device.type == 'cuda'), 'checkpoint native RNG device changed')
            with _device_scope(self.device):
                validate_rng(value['rng'])
        except (KeyError, TypeError, RuntimeError) as exc:
            raise ValueError('official Self-Forcing: invalid native RNG state') from exc

    def load_state_dict(self, value):
        self.validate_state_dict(value)
        self.model.load_state_dict(value['model'], strict=True)
        for name, tensor in plain_tensors(self.model).items():
            tensor.copy_(value['plain'][name])
        for role, optimizer in self.optimizers.items():
            optimizer.load_state_dict(deepcopy(value['optimizers'][role]))
        self.native_step = value['native_step']
        self.generator_updates, self.critic_updates = value['generator_updates'], value['critic_updates']
        self.optimizer_activity = deepcopy(value['optimizer_activity'])
        self.unconditional_dict = (None if value['unconditional_dict'] is None else
            {name: t.detach().to(device=self.device, copy=True) for name, t in value['unconditional_dict'].items()})
        self._rng = cpu_tree(value['rng'])
        self.generator_ema = None
        if value['ema'] is not None:
            self._initialize_ema()
            self.generator_ema.load_state_dict(value['ema'])
        self.ema_initialized_step, self.ema_updates = value['ema_initialized_step'], value['ema_updates']
        self._validate_live()

    def validate_checkpoint_progress(self, value, progress):
        cycle = value['native_step'] // self.ratio
        require(progress == dict(cycle=cycle, cursor=cycle, versions={'native_cycle': cycle}),
                'inner native clocks and outer Program cycle/cursor disagree')
