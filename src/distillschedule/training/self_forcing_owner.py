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


def _nested_code(function, *names):
    """Select an inspected source branch, without executing it or tracing RNG."""
    from types import CodeType
    code = function.__code__
    for name in names:
        found = [item for item in code.co_consts if isinstance(item, CodeType) and item.co_name == name]
        require(len(found) == 1, 'native checkpoint instrumentation source branch changed: ' + name)
        code = found[0]
    return code


def _checkpoint_observer(root, row):
    """Read-only hooks: count actual native checkpoint entry and replay entry.

    Wan calls a source-local custom_forward ONLY through torch checkpoint.
    Its immediate caller is checkpoint for the original execution, recompute_fn
    for backward. Match code objects, not names or a requested option. The hook
    never stores frames/tensors, replaces a forward, or changes checkpoint RNG.
    """
    import inspect
    import torch
    from torch.utils import checkpoint
    forward = type(root)._forward.__code__
    custom = _nested_code(type(root)._forward, 'create_custom_forward', 'custom_forward')
    initial = inspect.unwrap(checkpoint.checkpoint).__code__
    recompute = _nested_code(checkpoint._checkpoint_without_reentrant_generator, 'recompute_fn')

    def observe(module, inputs):
        frame = inspect.currentframe()
        try:
            frame = frame.f_back
            while frame is not None:
                if frame.f_code is custom:
                    caller = frame.f_back.f_code
                    require(caller in (initial, recompute), 'unrecognized native checkpoint invocation')
                    row['checkpoint_requests' if caller is initial else 'recompute_calls'] += 1
                    return
                if frame.f_code is forward:
                    row['direct_grad_calls'] += int(torch.is_grad_enabled())
                    return
                frame = frame.f_back
            raise ValueError('official Self-Forcing: block called outside the qualified native source branch')
        finally:
            del frame
    return observe


def stream_counts(args):
    return {'generator': 1, 'critic': args.dfake_gen_update_ratio}


def _hash(value):
    import torch
    from torch.distributed.tensor import DTensor
    # Each real rank checkpoints/validates its own immutable shard. The common
    # checkpoint publishes only after all ranks have validated their shards.
    if isinstance(value, DTensor):
        value = value.to_local()
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
        self._cache_lifetime = identity.get('native_cache_lifetime', 'native-v1')
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
        self._active_fake_checkpoint = None
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
        from .resident import active
        require(self.device.type != 'cuda' or (mode in ('reference', 'manual')
                and 'owner_execution' not in implementation) or active(self), CUDA_TRIAL_LIMIT)

    def _resident_trial_adapter(self):
        """Full host/frozen/plain state; never execute rollout for discovery."""
        from distillschedule.open_workload.resident_trial import (
            ResidentOwner, native_state_view, adam_growth, physical_attributes, cuda_growth)
        plain = plain_tensors(self.model)
        cache_bytes = self.args.batch_size * self.negative_embedding.numel() * self.negative_embedding.element_size()
        ema_bytes = sum(p.numel() * p.element_size() for p in self.model.generator.parameters())
        return ResidentOwner(self.model, self.optimizers,
            dict(native_state_view(self.model, self.optimizers), plain=plain,
                negative_embedding=self.negative_embedding, negative_prompt=self.config.negative_prompt,
                unconditional_dict=self.unconditional_dict, rng=self._rng,
                native_step=self.native_step, generator_updates=self.generator_updates,
                critic_updates=self.critic_updates, optimizer_activity=self.optimizer_activity,
                ema=None if self.generator_ema is None else self.generator_ema.shadow,
                ema_initialized_step=self.ema_initialized_step, ema_updates=self.ema_updates),
            aliases=dict(args=id(self.args), config=id(self.config), source=id(self.source),
                scheduler=id(self.model.scheduler), generator_scheduler=id(self.model.generator.scheduler),
                negative=id(self.negative_embedding), negative_prompt=id(self.config.negative_prompt),
                plain={name: (id(t), t.untyped_storage().data_ptr()) for name, t in plain.items()}),
            identity=dict(identity=self.identity, settings=vars(self.args), source=self.source.identity),
            growth_bytes=adam_growth(self.optimizers) + cache_bytes + ema_bytes, output_bytes=0,
            max_new_tensor_bytes=max(cache_bytes, *(p.numel() * p.element_size() for p in self.model.parameters())),
            physical=physical_attributes(self.model.modules(), ('gradient_checkpointing',)),
            retained={'ema': self.generator_ema},
            **cuda_growth(self.optimizers, host_bytes=ema_bytes, device_bytes=cache_bytes))

    def _resident_trial_cleanup(self):
        # An interrupted native forward can retain the complete rollout graph.
        self.model.inference_pipeline = None
        self._active_fake_checkpoint = None
        for parameter in self.model.parameters():
            parameter.grad = None

    def _resident_trial_restore(self, state, retained):
        from distillschedule.open_workload.resident_trial import restore_native_state
        restore_native_state(self.model, self.optimizers, state)
        self.native_step = state['native_step']
        self.generator_updates, self.critic_updates = state['generator_updates'], state['critic_updates']
        self.optimizer_activity = state['optimizer_activity']
        self.unconditional_dict, self._rng = state['unconditional_dict'], state['rng']
        if state['ema'] is None:
            self.generator_ema = None
        else:
            self.generator_ema = retained['ema']
            self.generator_ema.shadow = state['ema']
        self.ema_initialized_step, self.ema_updates = state['ema_initialized_step'], state['ema_updates']

    def _resident_trial_validate(self):
        require(not hasattr(self, '_resident_trial_poisoned'), 'resident trial owner is poisoned')
        self._validate_live()
        if self.device.type == 'cuda':
            require(len(self.execution_options()['options']) > 1,
                    'private CUDA resident trial requires source-qualified Fake-only checkpoint options')

    def execution_options(self):
        """Source-bound tiny capability; CUDA requires active private admission.

        The old G-OFF reference is still available but never compared/searched
        against G-ON: the pinned causal branches pass different cache arguments.
        Only noncausal Fake checkpoint OFF is qualified against explicit G/F-ON.
        """
        import inspect
        from pathlib import Path
        import torch
        from .self_forcing import runtime_identity, _native_config
        from distillschedule.open_workload.torch_backend import code_signature
        from distillschedule.workloads.best_known.registry import file_sha256
        from distillschedule.open_workload.resident_trial import cuda_trial_active
        from .resident import active
        self._validate_live()
        runtime = runtime_identity()
        required = dict(torch='2.10.0', diffusers='0.37.1', transformers='4.51.3', einops='0.8.1')
        stack = dict(runtime, torch=str(torch.__version__).split('+')[0])
        api = original(self.source.root)
        tiny = dict(model_type='t2v', patch_size=[1, 2, 2], text_len=8, in_dim=16, out_dim=16,
            dim=32, ffn_dim=64, freq_dim=16, text_dim=16, num_heads=2, num_layers=1,
            qk_norm=True, cross_attn_norm=True)
        architecture = {'generator': _native_config(api.causal.CausalWanModel,
                            dict(tiny, local_attn_size=-1, sink_size=0)),
                        **{role: _native_config(api.wan.WanModel, tiny) for role in ('real_score', 'fake_score')},
                        'vae': _native_config(api.vae.WanVAE_, dict(dim=4, z_dim=16, dim_mult=[1, 2, 2, 2],
                            num_res_blocks=1, attn_scales=[], temperal_downsample=[True, True, False], dropout=0.))}
        settings = dict(image_or_video_shape=[1, 21, 16, 4, 4], num_frame_per_block=3,
            dfake_gen_update_ratio=5, denoising_step_list=[1000, 750, 500, 250], warp_denoising_step=True,
            same_step_across_blocks=True, last_step_only=False, context_noise=0, fake_guidance_scale=0.,
            real_guidance_scale=3., timestep_shift=5., scheduler_shift=5., ts_schedule=False,
            ts_schedule_max=False, min_score_timestep=0)
        private_cuda = self.device.type == 'cuda' and (cuda_trial_active(self) or active(self))
        if self.device.type == 'cuda' and not private_cuda:
            status = 'UNAVAILABLE_CUDA_TRIAL_SNAPSHOT'
        elif not self.args.gradient_checkpointing:
            status = 'UNAVAILABLE_REFERENCE_POLICY'
        elif (not all(stack[name] == value for name, value in required.items())
                or runtime['cpu_policy']['threads'] != 1):
            status = 'UNAVAILABLE_RUNTIME'
        elif (self.identity['architecture'] != architecture
                or any(getattr(self.args, key) != value for key, value in settings.items())
                or self.identity['assets']['evidence_kind'] != 'RANDOM_TINY_ACTUAL_NATIVE_CLASSES_NOT_PRETRAINED'):
            status = 'UNAVAILABLE_TINY_PROFILE'
        else:
            status = 'AVAILABLE_TINY_PRIVATE_CUDA' if private_cuda else 'AVAILABLE_TINY_CPU'
        baseline = {'fake_gradient_checkpointing': self.args.gradient_checkpointing}
        descriptor = dict(schema_version='distillschedule.owner_execution.v1', baseline=baseline,
            options=[baseline], effect=dict(kind='native_checkpoint', enabled_key='fake_gradient_checkpointing',
                                           components=['fake']),
            identity=dict(source=deepcopy(self.source.identity), runtime=runtime,
                provider_file_sha256=file_sha256(Path(__file__)), generator_checkpoint=self.args.gradient_checkpointing,
                capability=dict(status=status, required_runtime=required, required_cpu_threads=1,
                    architecture=architecture, settings=settings, gpu='NOT_RUN', pretrained='NOT_RUN'),
                scope=('TINY_PRIVATE_CUDA1_FP32_NONCAUSAL_FAKE_ONLY_G_FIXED_SOURCE_IDENTITY' if private_cuda else
                       'TINY_CPU1_FP32_NONCAUSAL_FAKE_ONLY_G_FIXED_SOURCE_IDENTITY'),
                instrumentation='READ_ONLY_BLOCK_PRE_HOOK_NATIVE_CUSTOM_FORWARD_CHECKPOINT_AND_RECOMPUTE_CODE'))
        if status not in ('AVAILABLE_TINY_CPU', 'AVAILABLE_TINY_PRIVATE_CUDA'):
            return descriptor
        roots = (self.model.generator.model, self.model.real_score.model, self.model.fake_score.model)
        require(tuple(map(type, roots)) == (api.causal.CausalWanModel, api.wan.WanModel, api.wan.WanModel),
                'checkpoint requires actual unmodified native model classes')
        for root in roots:
            require(all(name not in vars(root) for name in ('forward', '_forward', '_set_gradient_checkpointing',
                        'enable_gradient_checkpointing', 'disable_gradient_checkpointing')),
                    'native checkpoint model methods changed')
        from torch.utils import checkpoint
        helper_files = {type(module).__module__: inspect.getsourcefile(type(module))
                        for root in roots for module in root.modules()}
        helper_files['torch.utils.checkpoint'] = checkpoint.__file__
        helper_files['torch.checkpoint_wrapper'] = inspect.getsourcefile(checkpoint.checkpoint)
        require(all(path is not None for path in helper_files.values()), 'checkpoint helpers require source files')
        descriptor['identity'].update(
            helper_files={name: file_sha256(Path(path)) for name, path in sorted(helper_files.items())},
            checkpoint_api=code_signature(inspect.unwrap(checkpoint.checkpoint)),
            recompute_api=code_signature(checkpoint._checkpoint_without_reentrant_generator))
        # Confirm the exact source branches the read-only observer will inspect.
        _nested_code(api.wan.WanModel._forward, 'create_custom_forward', 'custom_forward')
        _nested_code(checkpoint._checkpoint_without_reentrant_generator, 'recompute_fn')
        descriptor['options'].append({'fake_gradient_checkpointing': False})
        return descriptor

    @contextmanager
    def execution_scope(self, option, effects):
        descriptor = self.execution_options()
        require(self._active_fake_checkpoint is None and len(descriptor['options']) > 1
                and type(option) is dict and set(option) == {'fake_gradient_checkpointing'}
                and type(option['fake_gradient_checkpointing']) is bool and option in descriptor['options'],
                'unsupported native execution option; Generator checkpoint policy is immutable')
        root = self.model.fake_score.model
        saved = root.gradient_checkpointing
        row = dict(checkpoint_requests=0, recompute_calls=0, direct_grad_calls=0, disabled_verified=False)
        effects['components'] = {'fake': row}
        hooks = []
        try:
            root.gradient_checkpointing = self._active_fake_checkpoint = option['fake_gradient_checkpointing']
            observer = _checkpoint_observer(root, row)
            for block in root.blocks:
                hooks.append(block.register_forward_pre_hook(observer))
            yield
        finally:
            row['disabled_verified'] = (root.gradient_checkpointing is False and row['direct_grad_calls'] > 0
                and row['checkpoint_requests'] == row['recompute_calls'] == 0)
            for hook in hooks:
                hook.remove()
            root.gradient_checkpointing = saved
            self._active_fake_checkpoint = None

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

    def _validate_placement(self):
        import torch
        from .self_forcing import _admit_device
        require(_admit_device({'device': str(self.device)}) == self.model.device
                and self.identity['placement'] == dict(device=str(self.device), layout='SINGLE_DEVICE_RESIDENT',
                                                       distributed=False, amp=False)
                and self.model.args is self.args and self.dtype == self.model.dtype == torch.float32,
                'live native model/device/FP32 placement changed')
        require(all(type(t) in (torch.Tensor, torch.nn.Parameter) and t.device == self.device
                    and t.layout == torch.strided for t in self.runtime_tensors()),
                'plain dense single-device placement required; no distributed, offload or AMP layout')
        require(all(not t.is_floating_point() or t.dtype == torch.float32
                    for t in (*self.model.parameters(), *self.model.buffers()))
                and self.negative_embedding.dtype == torch.float32,
                'native module/condition FP32 dtype changed')

    def _validate_live(self):
        import torch
        from .self_forcing import runtime_identity
        require(self._cache_lifetime == self.identity.get('native_cache_lifetime', 'native-v1')
                and self._cache_lifetime in ('native-v1', 'release-after-commit-v1'),
                'native cache lifetime policy changed outside its source-bound profile')
        self._validate_placement()
        require(self.identity['runtime'] == runtime_identity(), 'live numerical runtime/precision policy changed')
        self.source.validate()
        require(type(self.native_step) is int and self.native_step % self.ratio == 0
                and 0 <= self.native_step <= self.args.train_iters
                and self.generator_updates == self.cycle and self.critic_updates == self.native_step,
                'owner is not at a complete native macrocycle')
        require(self.model.inference_pipeline is None, 'checkpoint requires discarded ephemeral rollout caches')
        require(vars(self.args) == self.identity['settings'], 'native settings mutated outside the owned lifecycle')
        fake_checkpoint = (self.args.gradient_checkpointing if self._active_fake_checkpoint is None
                           else self._active_fake_checkpoint)
        require(self.model.generator.model.gradient_checkpointing is self.args.gradient_checkpointing
                and self.model.fake_score.model.gradient_checkpointing is fake_checkpoint
                and self.model.real_score.model.gradient_checkpointing is False,
                'native checkpoint flags changed outside the Fake-only execution scope')
        require(all(p.grad is None for p in self.model.parameters()), 'checkpoint requires cleared gradients')
        require(self.masks == {name: p.requires_grad for name, p in self.model.named_parameters()},
                'native role parameter masks changed')
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
            self._validate_unconditional()

    def _validate_unconditional(self):
        import torch
        from distillschedule.data.self_forcing_inputs import validate_row
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

    def fwdbwd_one_step(self, batch, train_generator):
        native = original(self.source.root).trainer.Trainer.fwdbwd_one_step
        return native(self, batch, train_generator)

    def _forward_backward(self, row, generator):
        return self.fwdbwd_one_step({'prompts': list(row['prompt_embeds'].unbind(0))}, generator)

    def _before_pipeline_discard(self, cycle):
        pass

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
        if self._cache_lifetime == 'release-after-commit-v1':
            self._release_consumed_rollout_cache(role)

    def _release_consumed_rollout_cache(self, role):
        # Only reached after native fwdbwd_one_step (including backward/clip)
        # and the corresponding Adam/EMA commit have completed. The next
        # native rollout allocates its own fresh cache; no cache is reused or
        # released during causal attention, checkpoint replay or backward.
        from distillschedule.open_workload.torch_backend import storage_bytes
        pipeline = self.model.inference_pipeline
        require(pipeline is not None and pipeline.kv_cache1 is not None
                and pipeline.crossattn_cache is not None, 'missing completed native rollout cache')
        caches = [pipeline.kv_cache1, pipeline.kv_cache2, pipeline.crossattn_cache]
        released = storage_bytes(caches)
        require(released > 0, 'completed native cache has no owned storage')
        pipeline.kv_cache1 = pipeline.kv_cache2 = pipeline.crossattn_cache = None
        self._cache_cycle_effects['releases'][role] += 1
        self._cache_cycle_effects['released_owner_storage_bytes'] += released
        # This counts references relinquished by the owner, not allocator-free
        # bytes: an external diagnostic observer can still hold aliases.

    def run(self, feeds, *, cycle):
        require(not hasattr(self, '_resident_trial_poisoned'), 'resident trial owner is poisoned')
        self._validate_live()
        require(type(cycle) is int and cycle == self.cycle, 'native macrocycle reordered')
        require(self.native_step + self.ratio <= self.args.train_iters, 'native iteration budget exhausted')
        self.validate_feeds(feeds)
        if self._cache_lifetime == 'release-after-commit-v1':
            self._cache_cycle_effects = dict(policy=self._cache_lifetime, scope='THIS_COMPLETED_CYCLE',
                releases=dict(generator=0, critic=0), released_owner_storage_bytes=0,
                byte_scope='DEDUPLICATED_STORAGE_REFERENCES_RELINQUISHED_PER_COMMIT', helper_host_tensor_value_reads=0)
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
        self._before_pipeline_discard(cycle)
        self.model.inference_pipeline = None
        for optimizer in self.optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        self._validate_live()
        result = dict(iterations=rows, native_step=self.native_step, generator_updates=self.generator_updates,
            critic_updates=self.critic_updates, ema_updates=self.ema_updates,
            consumed={role: self.cycle * count for role, count in stream_counts(self.args).items()})
        if self._cache_lifetime == 'release-after-commit-v1':
            require(self._cache_cycle_effects['releases'] == dict(generator=1, critic=self.ratio),
                    'native consumed-cache release census changed')
            result['cache_lifetime_effects'] = deepcopy(self._cache_cycle_effects)
        return result

    def _cache_snapshot(self):
        pipeline = self.model.inference_pipeline
        return {'kv': pipeline.kv_cache1, 'crossattn': pipeline.crossattn_cache}

    def state_dict(self):
        require(not hasattr(self, '_resident_trial_poisoned'), 'resident trial owner is poisoned')
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
        self._validate_checkpoint_tensors(value)
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
        expected_condition = self._checkpoint_unconditional(negative_cpu) if step else None
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

    def _checkpoint_unconditional(self, negative_cpu):
        return {'prompt_embeds': negative_cpu.repeat(self.args.batch_size, 1, 1)}

    def _validate_checkpoint_tensors(self, value):
        import torch
        from torch.utils._pytree import tree_leaves
        require(all(type(t) is torch.Tensor and t.layout == torch.strided and not t.requires_grad
                    and (t.device.type == 'cpu' or t.device == self.device)
                    for t in tree_leaves(value) if isinstance(t, torch.Tensor)),
                'checkpoint requires dense CPU or owned-device staging tensors')

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
