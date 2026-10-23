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

SCHEMA = 'distillschedule.official_self_forcing_owner.v1'


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
        self.model, self.args, self.source = model, args, source
        self.identity = deepcopy(identity)
        self.negative_embedding = negative_embedding.detach().clone()
        # The original fwdbwd method calls a text encoder on a list. At this
        # explicit preencoded boundary each item is an actual native embedding;
        # the identity adapter does no tokenization or invented conditioning.
        self.config = SimpleNamespace(**{**vars(args), 'negative_prompt': self.negative_embedding[0]})
        self.device, self.dtype = model.device, model.dtype
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
        self.generator_ema = None
        self.ema_initialized_step = None
        if args.ema_start_step == 0 and args.ema_weight > 0:
            self._initialize_ema()
        self._rng = cpu_tree(rng_state())
        self.observer = None
        self._validate_live()

    @property
    def cycle(self):
        return self.native_step // self.ratio

    @property
    def step(self):
        return self.native_step

    def _event(self, name, value):
        if self.observer is not None:
            self.observer(name, value)

    @contextmanager
    def random_scope(self):
        outer = rng_state()
        restore_rng(self._rng)
        try:
            yield
        finally:
            self._rng = cpu_tree(rng_state())
            restore_rng(outer)

    def _ema_container(self):
        import torch
        # FSDP.summon_full_params is a no-op on this unsharded nn.Module tree.
        # This lets the original EMA_FSDP arithmetic and parameter naming run
        # verbatim on CPU rather than replacing its equations.
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
        if self.generator_ema is not None:
            yield from self.generator_ema.shadow.values()

    def _validate_live(self):
        import torch
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
        require(all(type(t) in (torch.Tensor, torch.nn.Parameter) and t.device.type == 'cpu'
                    and t.layout == torch.strided for t in self.runtime_tensors()),
                'plain dense CPU tensor ownership required; no distributed, offload or AMP layout')
        names = {id(p): name for name, p in self.model.named_parameters()}
        require(self.ownership == {role: [[names.get(id(p)) for p in group['params']]
                    for group in optimizer.param_groups] for role, optimizer in self.optimizers.items()},
                'optimizer role/name ownership changed')
        require(all(name.startswith('generator.') for group in self.ownership['generator'] for name in group)
                and all(name.startswith('fake_score.') for group in self.ownership['critic'] for name in group),
                'teacher or codec entered a trainable optimizer')
        model_state, native_plain = self.model.state_dict(), plain_tensors(self.model)
        require(all(_hash(model_state[name]) == digest for name, digest in self.frozen_hashes.items())
                and all(_hash(native_plain[name]) == digest for name, digest in self.plain_hashes.items()),
                'live frozen teacher/VAE or native non-buffer tensor changed')
        require(self.generator_ema is None or self.generator_ema.decay == self.args.ema_weight,
                'native EMA policy changed')

    def validate_feeds(self, feeds):
        from distillschedule.data.self_forcing_inputs import validate_row
        counts = stream_counts(self.args)
        require(isinstance(feeds, dict) and set(feeds) == set(counts), 'independent G/C input streams required')
        for role, count in counts.items():
            require(isinstance(feeds[role], list) and len(feeds[role]) == count,
                    'native prompt consumption changed for ' + role)
            for row in feeds[role]:
                validate_row(row, shape=(self.args.batch_size, self.model.generator.model.text_len,
                                         self.model.generator.model.text_dim))

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
        require(torch.equal(self.negative_embedding, value['negative_embedding']), 'negative conditioning changed')
        expected_condition = {'prompt_embeds': self.negative_embedding.repeat(self.args.batch_size, 1, 1)} if step else None
        _validate_layout(expected_condition, value['unconditional_dict'], 'native unconditional cache')
        require(expected_condition is None or torch.equal(expected_condition['prompt_embeds'],
                                                        value['unconditional_dict']['prompt_embeds']),
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
                        and state['step'].ndim == 0 and float(state['step']) == activity[ids[index]],
                        'native AdamW parameter update clock changed')
            _validate_optimizer_layout(optimizer, saved, 'Self-Forcing ' + role)
        require(all(bool(torch.isfinite(tensor).all()) for tensor in tree_leaves(value)
                    if isinstance(tensor, torch.Tensor) and (tensor.is_floating_point() or tensor.is_complex())),
                'nonfinite checkpoint tensor')
        try:
            require(set(value['rng']) == set(self._rng), 'native RNG fields changed')
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
        self.unconditional_dict = cpu_tree(value['unconditional_dict'])
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
