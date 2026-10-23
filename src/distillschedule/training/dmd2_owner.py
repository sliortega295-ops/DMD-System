"""Atomic ownership of original DMD2 native iterations and restart state.

Only the trainer lifecycle is adapted: the external SDUniModel/SDGuidance own
every denoising, score, classifier and loss calculation. A macrocycle performs
one G commit and ``ratio`` guidance commits, at the original native clocks.
"""
from contextlib import contextmanager
from copy import deepcopy
import hashlib

from .checkpoint import (rng_state, restore_rng, validate_rng,
                         _validate_layout, _validate_optimizer_layout)
from .sd import require
from distillschedule.workloads.best_known.registry import fingerprint

SCHEMA = 'distillschedule.official_dmd2_owner.v1'


def stream_counts(args):
    result = {'generator': 1}
    if args.dfake_gen_update_ratio > 1:
        result['guidance'] = args.dfake_gen_update_ratio - 1
    if args.denoising:
        result['denoising'] = args.dfake_gen_update_ratio
    if args.cls_on_clean_image:
        result['real'] = args.dfake_gen_update_ratio
    return result


def _clone(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone(item) for item in value]
    return deepcopy(value)


def _hash_tensor(value):
    import torch
    return hashlib.sha256(value.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy()).hexdigest()


def _tensor_identity(value):
    return dict(shape=list(value.shape), dtype=str(value.dtype), sha256=_hash_tensor(value))


class DMD2Owner:
    def __init__(self, model, args, source, uncond_embedding, *, identity, seed):
        import torch
        from diffusers.optimization import get_scheduler
        self.model, self.args, self.source = model, args, source
        self.uncond_embedding = uncond_embedding
        self.identity = deepcopy(identity)
        self.ratio, self.native_step = args.dfake_gen_update_ratio, 0
        self.generator_updates, self.guidance_updates = 0, 0
        self.optimizers = {
            'generator': torch.optim.AdamW([p for p in model.feedforward_model.parameters() if p.requires_grad],
                lr=args.generator_lr, betas=(0.9, 0.999), weight_decay=0.01),
            'guidance': torch.optim.AdamW([p for p in model.guidance_model.parameters() if p.requires_grad],
                lr=args.guidance_lr, betas=(0.9, 0.999), weight_decay=0.01),
        }
        self.schedulers = {name: get_scheduler('constant_with_warmup', optimizer=optimizer,
                num_warmup_steps=args.warmup_step, num_training_steps=args.train_iters)
            for name, optimizer in self.optimizers.items()}
        names = {id(p): name for name, p in model.named_parameters()}
        self.ownership = {role: [[names[id(p)] for p in group['params']] for group in optimizer.param_groups]
                          for role, optimizer in self.optimizers.items()}
        require(not any(name.startswith(('guidance_model.real_unet.', 'guidance_model.dummy_network.'))
                        for group in self.ownership['guidance'] for name in group),
                'teacher/dummy entered guidance optimizer')
        self.initial_masks = {name: p.requires_grad for name, p in model.named_parameters()}
        self.frozen_hashes = {name: _hash_tensor(parameter) for name, parameter in model.named_parameters()
                              if not parameter.requires_grad}
        self._settings_hash = fingerprint(vars(args))
        self._preprocessing_hash = fingerprint(self.identity['preprocessing_assets'])
        self._frozen_identity = self._frozen_tensors_identity()
        self._constants_hash = fingerprint(self._native_constants())
        self._rng = _clone(rng_state())
        self.observer = None  # Optional diagnostics; never serialized or used for math.
        self._validate_live()

    @property
    def cycle(self):
        return self.native_step // self.ratio

    @contextmanager
    def random_scope(self):
        """Restore the native stream inside (not from) Program's random scope."""
        outer = rng_state()
        restore_rng(self._rng)
        try:
            yield
        finally:
            self._rng = _clone(rng_state())
            restore_rng(outer)

    def _event(self, event, value):
        if self.observer is not None:
            self.observer(event, value)

    def execution_options(self):
        """Pure, source-bound CPU1/FP32 native UNet checkpoint capability.

        The bound args/metadata stay mathematical identity. Only a complete
        macrocycle's physical flags and supported Diffusers callbacks change.
        """
        import torch
        import inspect
        from pathlib import Path
        from diffusers import UNet2DConditionModel
        from .sd import runtime_identity
        from distillschedule.open_workload.torch_backend import code_signature
        from distillschedule.workloads.best_known.registry import file_sha256
        self._validate_live()
        runtime = runtime_identity()
        require(runtime == self.identity['runtime'], 'live numerical dependency runtime changed')
        required = dict(torch='2.10.0', diffusers='0.37.1', transformers='4.51.3', peft='0.17.1')
        stack = dict(runtime, torch=str(torch.__version__).split('+')[0])
        qualified = (all(stack[name] == version for name, version in required.items())
                     and not torch.distributed.is_initialized())
        baseline = {'gradient_checkpointing': self.args.gradient_checkpointing}
        descriptor = dict(schema_version='distillschedule.owner_execution.v1', baseline=baseline,
            options=[baseline], effect=dict(kind='native_checkpoint', enabled_key='gradient_checkpointing',
                                           components=['generator', 'fake']),
            identity=dict(source=deepcopy(self.source.identity), runtime=runtime,
                provider_file_sha256=file_sha256(Path(__file__)),
                native_classes={name: code_signature(type(value)) for name, value in (
                    ('model', self.model), ('guidance', self.model.guidance_model))},
                capability=dict(status='AVAILABLE' if qualified else 'UNAVAILABLE_RUNTIME', required_runtime=required),
                scope='CPU1_FP32_NATIVE_UNETS_ONLY_NO_TEACHER_CLIP_VAE'))
        # Optional checkpoint qualification must not narrow the preexisting
        # reference factory's runtime admission. No override is offered here.
        if not qualified:
            return descriptor
        require(all(t.device.type == 'cpu' and (not t.is_floating_point() or t.dtype == torch.float32)
                    for t in self.runtime_tensors()), 'native checkpoint requires CPU1/FP32 tensors')
        roots = (self.model.feedforward_model, self.model.guidance_model.fake_unet)
        for root in roots:
            require(type(root) is UNet2DConditionModel and root._supports_gradient_checkpointing
                    and all(name not in vars(root) for name in (
                        'enable_gradient_checkpointing', 'disable_gradient_checkpointing', '_set_gradient_checkpointing'))
                    and any(hasattr(module, 'gradient_checkpointing') for module in root.modules())
                    and all(type(module.gradient_checkpointing) is bool for module in root.modules()
                            if hasattr(module, 'gradient_checkpointing')),
                    'native checkpoint requires unmodified checkpoint-capable actual UNets')
        from . import checkpoint as checkpoint_state, sd
        helper_files = {type(module).__module__: inspect.getsourcefile(type(module))
                        for root in roots for module in root.modules()}
        helper_files.update({module.__name__: module.__file__ for module in (checkpoint_state, sd)})
        helper_files[type(self.source).__module__] = inspect.getsourcefile(type(self.source))
        helper_files['torch.utils.checkpoint'] = inspect.getsourcefile(inspect.unwrap(torch.utils.checkpoint.checkpoint))
        helper_files['torch.checkpoint_wrapper'] = inspect.getsourcefile(torch.utils.checkpoint.checkpoint)
        require(all(path is not None for path in helper_files.values()),
                'native checkpoint helpers require inspectable source files')
        descriptor['options'].append({'gradient_checkpointing': not self.args.gradient_checkpointing})
        descriptor['identity'].update(
            helper_files={name: file_sha256(Path(path)) for name, path in sorted(helper_files.items())},
            checkpoint_api={name: code_signature(getattr(UNet2DConditionModel, name)) for name in (
                'enable_gradient_checkpointing', 'disable_gradient_checkpointing', '_set_gradient_checkpointing')},
            torch_checkpoint=code_signature(inspect.unwrap(torch.utils.checkpoint.checkpoint)))
        return descriptor

    @contextmanager
    def execution_scope(self, option, effects):
        """Retain original native toggles, observe real requests and callable replay.

        Upstream re-enables checkpointing without a callback on non-G turns and
        every fake turn. Scope the supported enable API too, so those original
        calls reinstall our measured, numerically identical native callback.
        Never wrap or replace a UNet forward, change masks, modes, args or RNG.
        """
        import torch
        descriptor = self.execution_options()
        require(len(descriptor['options']) > 1 and type(option) is dict and set(option) == {'gradient_checkpointing'}
                and type(option['gradient_checkpointing']) is bool and option in descriptor['options'],
                'unsupported native execution option')
        enabled = option['gradient_checkpointing']
        roots = {'generator': self.model.feedforward_model, 'fake': self.model.guidance_model.fake_unet}
        flags = [(owner, owner.gradient_checkpointing) for owner in (self.model, self.model.guidance_model)]
        attributes = ('gradient_checkpointing', '_gradient_checkpointing_func', 'enable_gradient_checkpointing')
        saved = [(module, {key: (key in vars(module), vars(module).get(key)) for key in attributes})
                 for root in roots.values() for module in root.modules()]
        effects['components'] = {name: dict(checkpoint_requests=0, recompute_calls=0, disabled_verified=False)
                                 for name in roots}

        def measured_enable(root, row):
            original = root.enable_gradient_checkpointing
            def checkpoint(module, *args):
                row['checkpoint_requests'] += 1
                invocations = 0
                def call(*values):
                    nonlocal invocations
                    invocations += 1
                    if invocations > 1:
                        row['recompute_calls'] += 1
                    return module.__call__(*values)
                # Exactly Diffusers0.37's default callback and RNG behavior.
                return torch.utils.checkpoint.checkpoint(call, *args, use_reentrant=False)
            def enable(gradient_checkpointing_func=None):
                require(gradient_checkpointing_func is None,
                        'nested native checkpoint callback override is not admitted')
                return original(gradient_checkpointing_func=checkpoint)
            return enable

        try:
            for name, root in roots.items():
                root.enable_gradient_checkpointing = measured_enable(root, effects['components'][name])
                root.disable_gradient_checkpointing()
            self.model.gradient_checkpointing = self.model.guidance_model.gradient_checkpointing = enabled
            if enabled:
                roots['generator'].enable_gradient_checkpointing()
            # Fake remains disabled until its original compute_loss_fake enables
            # it, and the original disable at the end of that forward remains.
            yield
        finally:
            for name, root in roots.items():
                effects['components'][name]['disabled_verified'] = all(
                    not module.gradient_checkpointing for module in root.modules()
                    if hasattr(module, 'gradient_checkpointing'))
            for module, values in reversed(saved):
                for key, (present, value) in values.items():
                    if present:
                        setattr(module, key, value)
                    elif key in vars(module):
                        delattr(module, key)
            for owner, value in flags:
                owner.gradient_checkpointing = value

    def runtime_tensors(self):
        """Device-owned tensors, excluding host-only RNG/Adam step metadata.

        The common backend consumes this live-tensor seam, not state_dict's
        cloned host metadata. This does not admit CUDA: native placement and
        checkpoint restore remain validated only on CPU1/FP32.
        """
        yield from self.model.parameters()
        yield from self.model.buffers()
        yield from self._plain_tensors().values()
        for parameter in self.model.parameters():
            if parameter.grad is not None:
                yield parameter.grad
        for optimizer in self.optimizers.values():
            for state in optimizer.state.values():
                for name, tensor in state.items():
                    if name != 'step':
                        yield tensor

    def _plain_tensors(self):
        import torch
        result = {}
        for prefix, owner in (('model', self.model), ('scheduler', self.model.noise_scheduler)):
            result.update({prefix + '.' + name: value for name, value in vars(owner).items()
                           if isinstance(value, torch.Tensor)})
        if self.uncond_embedding is not None:
            result['uncond_embedding'] = self.uncond_embedding
        return result

    def _frozen_tensors_identity(self):
        # Initial masks, not current masks: upstream deliberately toggles the
        # entire guidance module to requires_grad=True after the first G turn.
        # Those Teacher tensors remain immutable and outside the optimizer.
        tensors = dict(self.model.named_parameters())
        result = {name: _tensor_identity(tensors[name]) for name in self.frozen_hashes}
        for name, value in self.model.named_buffers():
            if not name.startswith(('feedforward_model.', 'guidance_model.fake_unet.',
                                    'guidance_model.cls_pred_branch.')):
                result[name] = _tensor_identity(value)
        return result

    def _native_constants(self):
        from .sd import _plain
        # Bound args.gradient_checkpointing remains mathematical identity. The
        # native model's actual flags/callbacks are physical execution state;
        # admission here must not preempt a separately scoped capability owner.
        model_fields = ('num_train_timesteps', 'num_visuals', 'conditioning_timestep', 'use_fp16',
            'backward_simulation', 'cls_on_clean_image', 'denoising',
            'denoising_timestep', 'num_denoising_step', 'timestep_interval', 'sdxl', 'not_sdxl_vae')
        guidance_fields = ('gan_alone', 'num_train_timesteps', 'min_step', 'max_step',
            'real_guidance_scale', 'fake_guidance_scale', 'use_fp16', 'cls_on_clean_image',
            'gen_cls_loss', 'sdxl', 'diffusion_gan', 'diffusion_gan_max_timestep')
        return dict(model={name: getattr(self.model, name) for name in model_fields},
            guidance={name: getattr(self.model.guidance_model, name) for name in guidance_fields},
            tensors={name: _tensor_identity(value) for name, value in self._plain_tensors().items()},
            scheduler=_plain(dict(self.model.noise_scheduler.config)),
            scheduler_fields={name: value for name, value in vars(self.model.noise_scheduler).items()
                if not name.startswith('_') and type(value) in (str, bool, int, float, type(None))},
            vae=_plain(dict(self.model.vae.config)))

    def _masks(self, native_step):
        return {name: (True if native_step > 0 and name.startswith('guidance_model.') else value)
                for name, value in self.initial_masks.items()}

    def _validate_live(self):
        from distillschedule.data.sd_inputs import preprocessing_assets
        self.source.validate()
        require(fingerprint(vars(self.args)) == self._settings_hash
                and fingerprint(self.identity['settings']) == self._settings_hash
                and self.model.args is self.args and self.model.guidance_model.args is self.args
                and self.ratio == self.args.dfake_gen_update_ratio, 'live mathematical settings changed')
        require(fingerprint(self.identity['preprocessing_assets']) == self._preprocessing_hash
                and self.identity['preprocessing_assets'] == preprocessing_assets(self.identity['bundle']['files'],
                    variant='sdxl' if self.args.sdxl else 'sd15', images=True),
                'live tokenizer/VAE preprocessing binding changed')
        require(self._frozen_tensors_identity() == self._frozen_identity,
                'live frozen Teacher/CLIP/VAE/base-LoRA tensors changed')
        require(self.model.noise_scheduler is self.model.guidance_model.scheduler
                and fingerprint(self._native_constants()) == self._constants_hash,
                'live native conditioning/scheduler constants changed')
        require(self.native_step % self.ratio == 0 and self.generator_updates == self.cycle
                and self.guidance_updates == self.native_step, 'not a complete native macrocycle')
        names = {id(p): name for name, p in self.model.named_parameters()}
        require(self.ownership == {role: [[names.get(id(p)) for p in group['params']]
                for group in optimizer.param_groups] for role, optimizer in self.optimizers.items()},
                'optimizer parameter/name ownership changed')
        require(all(p.grad is None for p in self.model.parameters()), 'checkpoint requires cleared native gradients')
        require({name: p.requires_grad for name, p in self.model.named_parameters()} == self._masks(self.native_step),
                'native teacher/fake/generator masks changed')

    def validate_feeds(self, feeds):
        from distillschedule.data.sd_inputs import validate_row
        counts = stream_counts(self.args)
        require(isinstance(feeds, dict) and set(feeds) == set(counts), 'native input streams changed')
        for name, count in counts.items():
            require(isinstance(feeds[name], list) and len(feeds[name]) == count,
                    'wrong samples_per_cycle for stream ' + name)
            for row in feeds[name]:
                validate_row(row, sdxl=self.args.sdxl, model=self.model,
                             binding=self.identity['preprocessing_assets'],
                             images=name == 'real' or (name == 'denoising' and not self.args.backward_simulation))

    def run(self, feeds, *, cycle):
        self._validate_live()
        require(type(cycle) is int and cycle == self.cycle, 'native macrocycle reordered')
        self.validate_feeds(feeds)
        iterations = []
        with self.random_scope():
            for offset in range(self.ratio):
                prompt = feeds['generator'][0] if offset == 0 else feeds['guidance'][offset - 1]
                denoising = feeds['denoising'][offset] if self.args.denoising else None
                real = feeds['real'][offset] if self.args.cls_on_clean_image else None
                iterations.append(self._native_iteration(_clone(prompt), _clone(denoising), _clone(real)))
        self._validate_live()
        return {'iterations': iterations, 'native_step': self.native_step,
                'generator_updates': self.generator_updates, 'guidance_updates': self.guidance_updates,
                'scheduler_steps': {name: scheduler.last_epoch for name, scheduler in self.schedulers.items()},
                'ema_updates': 0, 'consumed': {name: self.cycle * count
                                             for name, count in stream_counts(self.args).items()}}

    def _native_iteration(self, prompt, denoising_dict, real_train_dict):
        """Original Trainer.train_one_step update ordering (no logger/loader)."""
        import torch
        model, args = self.model, self.args
        model.train()
        noise = torch.randn(args.batch_size, args.latent_channel, args.latent_resolution,
                            args.latent_resolution, device=model.accelerator.device)
        compute_generator = self.native_step % self.ratio == 0
        text = prompt if args.sdxl else prompt['text_input_ids_one'].squeeze(1)
        uncond = None if args.sdxl else self.uncond_embedding.repeat(len(text), 1, 1)
        generator_loss_dict, generator_log_dict = model(noise, text, uncond,
            visual=False, denoising_dict=denoising_dict, real_train_dict=real_train_dict,
            compute_generator_gradient=compute_generator, generator_turn=True, guidance_turn=False)
        cached = generator_log_dict['guidance_data_dict']
        self._event('pre_generator_commit_cache', cached)
        norms = {}
        if compute_generator:
            generator_loss = 0.0
            if not args.gan_alone:
                generator_loss += generator_loss_dict['loss_dm'] * args.dm_loss_weight
            if args.cls_on_clean_image and args.gen_cls_loss:
                generator_loss += generator_loss_dict['gen_cls_loss'] * args.gen_cls_loss_weight
            generator_loss.backward()
            self._event('generator_gradients', {name: p.grad for name, p in model.feedforward_model.named_parameters()})
            norms['generator_grad_norm'] = float(torch.nn.utils.clip_grad_norm_(
                model.feedforward_model.parameters(), args.max_grad_norm))
            self.optimizers['generator'].step()
            self.generator_updates += 1
            self.optimizers['generator'].zero_grad()
            self.optimizers['guidance'].zero_grad()
            self._event('generator_committed', model.feedforward_model.state_dict())
        # Both LR schedulers advance every native iteration, even without G.
        self.schedulers['generator'].step()
        self._event('guidance_input_cache', cached)
        guidance_loss_dict, _ = model(noise, text, uncond, visual=False,
            denoising_dict=denoising_dict, real_train_dict=real_train_dict,
            compute_generator_gradient=compute_generator, generator_turn=False,
            guidance_turn=True, guidance_data_dict=cached)
        guidance_loss = 0
        guidance_loss += guidance_loss_dict['loss_fake_mean']
        if args.cls_on_clean_image:
            guidance_loss += guidance_loss_dict['guidance_cls_loss'] * args.guidance_cls_loss_weight
        guidance_loss.backward()
        self._event('guidance_gradients', {name: p.grad for name, p in model.guidance_model.named_parameters()})
        norms['guidance_grad_norm'] = float(torch.nn.utils.clip_grad_norm_(
            model.guidance_model.parameters(), args.max_grad_norm))
        self.optimizers['guidance'].step()
        self.guidance_updates += 1
        self.optimizers['guidance'].zero_grad()
        self.optimizers['generator'].zero_grad()
        self.schedulers['guidance'].step()
        losses = {name: float(value.detach()) for name, value in
                  {**generator_loss_dict, **guidance_loss_dict}.items()}
        result = {'native_step': self.native_step, 'generator_turn': compute_generator,
                  'losses': losses, **norms,
                  'denoising_timesteps': generator_log_dict['denoising_timestep'].detach().cpu().tolist()}
        self.native_step += 1
        self._event('iteration_complete', result)
        return result

    def state_dict(self):
        self._validate_live()
        return dict(schema_version=SCHEMA, identity=deepcopy(self.identity), native_step=self.native_step,
            generator_updates=self.generator_updates, guidance_updates=self.guidance_updates,
            consumed={name: self.cycle * count for name, count in stream_counts(self.args).items()},
            model=self.model.state_dict(), training={name: child.training for name, child in self.model.named_modules()},
            requires_grad={name: p.requires_grad for name, p in self.model.named_parameters()},
            optimizer_ownership=deepcopy(self.ownership),
            optimizers={name: value.state_dict() for name, value in self.optimizers.items()},
            schedulers={name: value.state_dict() for name, value in self.schedulers.items()},
            uncond_embedding=self.uncond_embedding, rng=_clone(self._rng))

    def validate_state_dict(self, value):
        import torch
        from torch.utils._pytree import tree_leaves
        current = self.state_dict()
        require(isinstance(value, dict) and value.keys() == current.keys()
                and value['schema_version'] == SCHEMA and value['identity'] == self.identity,
                'checkpoint source/model/settings identity changed')
        step = value['native_step']
        require(type(step) is int and step >= 0 and step % self.ratio == 0 and step <= self.args.train_iters,
                'invalid checkpoint native step')
        cycle = step // self.ratio
        require(value['generator_updates'] == cycle and value['guidance_updates'] == step
                and type(value['generator_updates']) is int and type(value['guidance_updates']) is int
                and value['consumed'] == {name: cycle * count for name, count in stream_counts(self.args).items()},
                'checkpoint native update/data clocks changed')
        require(value['requires_grad'] == self._masks(step)
                and value['training'].keys() == current['training'].keys()
                and all(type(flag) is bool for flag in value['training'].values()),
                'checkpoint native parameter masks/module modes changed')
        _validate_layout(current['model'], value['model'], 'DMD2 model')
        require(all(_hash_tensor(value['model'][name]) == digest for name, digest in self.frozen_hashes.items()),
                'frozen teacher/CLIP/VAE/base-LoRA tensor changed')
        _validate_layout(current['uncond_embedding'], value['uncond_embedding'], 'DMD2 unconditional CLIP')
        require(self.uncond_embedding is None or torch.equal(self.uncond_embedding, value['uncond_embedding']),
                'empty-prompt CLIP conditioning changed')
        require(value['optimizer_ownership'] == self.ownership and set(value['optimizers']) == set(self.optimizers)
                and set(value['schedulers']) == set(self.schedulers), 'checkpoint optimizer ownership changed')
        for role, optimizer in self.optimizers.items():
            saved = value['optimizers'][role]
            expected_groups = current['optimizers'][role]['param_groups']
            require(isinstance(saved, dict) and set(saved) == {'state', 'param_groups'}
                    and len(saved['param_groups']) == len(expected_groups), 'invalid AdamW checkpoint groups')
            for actual, expected in zip(saved['param_groups'], expected_groups, strict=True):
                require(actual.keys() == expected.keys() and actual['params'] == expected['params']
                        and all(actual[key] == expected[key] for key in expected
                                if key not in ('lr', 'params')), 'AdamW group policy/name layout changed')
            slots = {index for group in expected_groups for index in group['params']}
            require(set(saved['state']) == (slots if step else set()), 'invalid/missing AdamW state')
            for state in saved['state'].values():
                require(set(state) == {'step', 'exp_avg', 'exp_avg_sq'}
                        and isinstance(state['step'], torch.Tensor) and state['step'].numel() == 1
                        and state['step'].dtype == torch.float32
                        and float(state['step']) == (cycle if role == 'generator' else step),
                        'AdamW moment/native update clock changed')
            _validate_optimizer_layout(optimizer, saved, 'DMD2 ' + role)
            scheduler = value['schedulers'][role]
            require(scheduler.keys() == current['schedulers'][role].keys()
                    and type(scheduler['last_epoch']) is int and scheduler['last_epoch'] == step
                    and type(scheduler['_step_count']) is int and scheduler['_step_count'] == step + 1
                    and all(scheduler[key] == current['schedulers'][role][key]
                            for key in scheduler if key not in ('last_epoch', '_step_count', '_last_lr')),
                    'both LR schedulers must track every native iteration')
            warmup = self.args.warmup_step
            base = self.args.generator_lr if role == 'generator' else self.args.guidance_lr
            lr = base * (min(float(step) / float(max(1, warmup)), 1.) if step < warmup else 1.)
            require(scheduler['base_lrs'] == [base] and scheduler['_last_lr'] == [lr]
                    and all(group['lr'] == lr for group in saved['param_groups']), 'LR scheduler recurrence changed')
        require(all(bool(torch.isfinite(tensor).all()) for tensor in tree_leaves(value)
                    if isinstance(tensor, torch.Tensor) and tensor.is_floating_point()),
                'nonfinite checkpoint tensor')
        validate_rng(value['rng'])

    def load_state_dict(self, value):
        self.validate_state_dict(value)  # Validate everything before mutating live tensors.
        self.model.load_state_dict(value['model'], strict=True)
        for name, child in self.model.named_modules():
            child.training = value['training'][name]
        for name, parameter in self.model.named_parameters():
            parameter.requires_grad_(value['requires_grad'][name])
            parameter.grad = None
        for name, optimizer in self.optimizers.items():
            optimizer.load_state_dict(deepcopy(value['optimizers'][name]))
            self.schedulers[name].load_state_dict(deepcopy(value['schedulers'][name]))
        self.native_step = value['native_step']
        self.generator_updates, self.guidance_updates = value['generator_updates'], value['guidance_updates']
        self.uncond_embedding = _clone(value['uncond_embedding'])
        self._rng = _clone(value['rng'])
        self._validate_live()

    def validate_checkpoint_progress(self, value, progress):
        cycle = value['native_step'] // self.ratio
        require(progress == {'cycle': cycle, 'cursor': cycle, 'versions': {'native_cycle': cycle}},
                'inner native clocks and outer Program cycle/cursor disagree')
