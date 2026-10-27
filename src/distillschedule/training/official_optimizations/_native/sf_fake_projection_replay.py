"""Measured implementation helper, relocated from r4_researchboost_transfer/scripts/sf_fake_projection_replay.py.

The public owner validates capabilities and lifecycle; this module does not
launch experiments or claim framework GPU acceptance. Source lock records all
original and relocated executable definitions.
"""
import hashlib
import inspect
from pathlib import Path

def require(value, reason):
    if not value:
        raise ValueError('SF_CRITIC_ATTENTION_REPLAY:' + reason)

class CriticAttentionReplay:

    def __init__(self, trainer, *, cpu_test_only=False, raw_op=None, max_logical_cache_bytes=4 * 2 ** 30, max_ffn_cache_bytes=2 * 2 ** 30):
        import torch
        import torch.utils.checkpoint as checkpoint
        from wan.modules.model import WanModel
        require(hashlib.sha256(Path(inspect.getsourcefile(WanModel)).read_bytes()).hexdigest() == 'f15214485a5297e6f9e79684bbad80acbcdc88c3272596b256a739a734529449', 'PINNED_NATIVE_MODEL_SOURCE')
        self.trainer = trainer
        self.cpu = cpu_test_only
        require(type(cpu_test_only) is bool and max_logical_cache_bytes > 0, 'EXPLICIT_CAPACITY_POLICY')
        if not self.cpu:
            require(raw_op is None and str(torch.__version__) == '2.10.0+cu128', 'PINNED_TARGET_ONLY')
            from flash_attn import flash_attn_interface as interface
            raw_op = interface._flash_attn_forward._opoverload
            require(raw_op._schema.name.endswith('::_flash_attn_forward'), 'RAW_OP_NAMESPACE')
        else:
            require(raw_op is not None and (not torch.cuda.is_initialized()), 'EXPLICIT_CPU_SURROGATE_ONLY')
        self.op = raw_op
        require(isinstance(raw_op, torch._ops.OpOverload), 'REAL_DISPATCH_OPERATOR_REQUIRED')
        require(len(raw_op._schema.returns) == 4 and len(raw_op._schema.arguments) == 11, 'FA2_RAW_ABI')
        self.checkpoint = checkpoint
        self.original = checkpoint.checkpoint
        self.blocks = {id(block): index for index, block in enumerate(trainer.model.fake_score.model.blocks)}
        self.version = 0
        self.pending = []
        self.limit = max_logical_cache_bytes
        self.ffn_limit = max_ffn_cache_bytes
        self.ffn_live_bytes = 0
        self.projection_live_bytes = 0
        self.projection_limit = 4 * 2 ** 30
        self.down = torch.ops.aten.addmm.default
        self.width = trainer.model.fake_score.model.dim
        self.ffn_width = trainer.model.fake_score.model.ffn_dim
        require(self.cpu or (self.width, self.ffn_width) == (1536, 8960), 'PINNED_NATIVE_FFN_GEOMETRY')
        self.live_bytes = 0
        self.record = dict(mechanism='PINNED_FA2_OUTPUT_LSE_SELECTIVE_CHECKPOINT_REUSE', checkpoint_calls=0, cache_writes=0, cache_reads=0, fake_commits=0, cache_logical_peak_bytes=0, max_logical_cache_bytes=max_logical_cache_bytes, saved_raw_forward_calls=0, contract_failures=0, rng_draws_added=0, tensor_host_value_reads=0, backward_implementation_changed=False, other_roles_checkpoint_changed=False, raw_op=str(raw_op), cpu_surrogate_only=self.cpu)
        self.record.update(ffn_replay_mechanism='FAKE_FFN_DOWN_OUTPUT_SELECTIVE_CHECKPOINT_REUSE', ffn_cache_writes=0, ffn_cache_reads=0, ffn_saved_addmm_calls=0, ffn_cache_logical_peak_bytes=0, combined_cache_logical_peak_bytes=0, max_ffn_cache_logical_bytes=self.ffn_limit, max_combined_cache_logical_bytes=self.limit + self.ffn_limit, cache_byte_scope='FA2_ABI_BYTES_ONLY_FFN_AND_COMBINED_BYTES_SEPARATE')
        self.record.update(projection_cache_writes=0, projection_cache_reads=0, projection_saved_addmm_calls=0, projection_cache_logical_peak_bytes=0, all_cache_logical_peak_bytes=0, max_projection_logical_bytes=self.projection_limit, max_all_cache_logical_bytes=self.limit + self.ffn_limit + self.projection_limit, projection_roles={'o': 0, 'co': 0}, projection_mechanism='EXACT_FAKE_O_CO_OUTPUT_SAC')

        def native_checkpoint(function, *args, **kwargs):
            modules = [cell.cell_contents for cell in getattr(function, '__closure__', None) or () if id(cell.cell_contents) in self.blocks]
            if not modules:
                return self.original(function, *args, **kwargs)
            require(len(modules) == 1 and torch.is_grad_enabled(), 'FAKE_BLOCK_SCOPE')
            require(kwargs.get('use_reentrant') is False and 'context_fn' not in kwargs, 'ORIGINAL_NONREENTRANT_CHECKPOINT_REQUIRED')
            state = dict(block=self.blocks[id(modules[0])], version=self.version, step=int(trainer.step), writes=0, reads=0, slots=[], ffn_writes=0, ffn_reads=0, ffn_slots=[], square_counts=[0, 0], projection_slots=[], projection_writes=0, projection_reads=0)
            self.pending.append(state)
            self.record['checkpoint_calls'] += 1

            def policy(context, operation, *op_args, **op_kwargs):
                if operation is self.down and len(op_args) == 3:
                    bias, x, weight = op_args[:3]
                    token_rows = args[0].shape[1]
                    if x.ndim == weight.ndim == 2 and tuple(weight.shape) == (self.width, self.width) and (x.shape[1] == self.width) and (x.shape[0] == token_rows):
                        index = state['square_counts'][int(context.is_recompute)]
                        state['square_counts'][int(context.is_recompute)] += 1
                        require(index < 6, 'NATIVE_SQUARE_GEMM_ORDER')
                        if index in (3, 5):
                            require(state['version'] == self.version and state['step'] == int(trainer.step), 'PROJECTION_VERSION_OR_STEP')
                            require(x.dtype == weight.dtype == bias.dtype == torch.bfloat16 and tuple(bias.shape) == (self.width,) and (x.device == weight.device == bias.device) and (op_kwargs.get('alpha', 1) == op_kwargs.get('beta', 1) == 1), 'PROJECTION_NATIVE_ABI')
                            require(self.cpu or x.shape[0] == 19320, 'PROJECTION_NATIVE_TOKENS')
                            role = 'o' if index == 3 else 'co'
                            metadata = (role,) + tuple(((tuple(t.shape), tuple(t.stride()), str(t.dtype), str(t.device)) for t in (bias, x, weight)))
                            size = x.shape[0] * self.width * x.element_size()
                            if context.is_recompute:
                                slot = state['projection_reads']
                                require(slot < len(state['projection_slots']) and state['projection_slots'][slot] == (metadata, size), 'PROJECTION_REPLAY_IDENTITY')
                                state['projection_reads'] += 1
                                self.record['projection_cache_reads'] += 1
                                self.record['projection_saved_addmm_calls'] += 1
                                self.projection_live_bytes -= size
                                require(self.projection_live_bytes >= 0, 'PROJECTION_NEGATIVE_BYTES')
                            else:
                                require(state['projection_reads'] == 0, 'PROJECTION_FORWARD_AFTER_REPLAY')
                                state['projection_slots'].append((metadata, size))
                                state['projection_writes'] += 1
                                self.record['projection_cache_writes'] += 1
                                self.record['projection_roles'][role] += 1
                                self.projection_live_bytes += size
                                self.record['projection_cache_logical_peak_bytes'] = max(self.record['projection_cache_logical_peak_bytes'], self.projection_live_bytes)
                                self.record['all_cache_logical_peak_bytes'] = max(self.record['all_cache_logical_peak_bytes'], self.projection_live_bytes + self.live_bytes + self.ffn_live_bytes)
                                require(self.projection_live_bytes <= self.projection_limit, 'PROJECTION_CAPACITY')
                            return checkpoint.CheckpointPolicy.MUST_SAVE
                    if x.ndim == weight.ndim == 2 and x.shape[1] == self.ffn_width and (tuple(weight.shape) == (self.ffn_width, self.width)):
                        require(state['version'] == self.version and state['step'] == int(trainer.step), 'FFN_VERSION_OR_STEP_CHANGED')
                        require(x.dtype == weight.dtype == bias.dtype == torch.bfloat16 and tuple(bias.shape) == (self.width,) and (x.device == weight.device == bias.device) and (op_kwargs.get('alpha', 1) == op_kwargs.get('beta', 1) == 1), 'FFN_NATIVE_DTYPE_BIAS_OR_SCALE')
                        require(self.cpu or x.shape[0] == 19320, 'FFN_NATIVE_TOKEN_LENGTH')
                        metadata = tuple(((tuple(t.shape), tuple(t.stride()), str(t.dtype), str(t.device)) for t in (bias, x, weight)))
                        size = x.shape[0] * self.width * x.element_size()
                        if context.is_recompute:
                            slot = state['ffn_reads']
                            require(slot < len(state['ffn_slots']) and state['ffn_slots'][slot] == (metadata, size), 'FFN_REPLAY_METADATA_OR_ORDER')
                            state['ffn_reads'] += 1
                            self.record['ffn_cache_reads'] += 1
                            self.record['ffn_saved_addmm_calls'] += 1
                            self.ffn_live_bytes -= size
                            require(self.ffn_live_bytes >= 0, 'FFN_NEGATIVE_LOGICAL_CACHE')
                        else:
                            require(state['ffn_reads'] == 0, 'FFN_FORWARD_AFTER_REPLAY')
                            state['ffn_slots'].append((metadata, size))
                            state['ffn_writes'] += 1
                            self.record['ffn_cache_writes'] += 1
                            self.ffn_live_bytes += size
                            self.record['ffn_cache_logical_peak_bytes'] = max(self.record['ffn_cache_logical_peak_bytes'], self.ffn_live_bytes)
                            self.record['combined_cache_logical_peak_bytes'] = max(self.record['combined_cache_logical_peak_bytes'], self.live_bytes + self.ffn_live_bytes)
                            self.record['all_cache_logical_peak_bytes'] = max(self.record['all_cache_logical_peak_bytes'], self.projection_live_bytes + self.live_bytes + self.ffn_live_bytes)
                            require(self.ffn_live_bytes <= self.ffn_limit and self.live_bytes + self.ffn_live_bytes <= self.limit + self.ffn_limit, 'FFN_COMBINED_CAPACITY_EXCEEDED')
                        return checkpoint.CheckpointPolicy.MUST_SAVE
                if operation is not self.op:
                    return checkpoint.CheckpointPolicy.PREFER_RECOMPUTE
                require(state['version'] == self.version and state['step'] == int(trainer.step), 'CACHE_VERSION_OR_STEP_CHANGED')
                arguments = dict(zip((a.name for a in self.op._schema.arguments), op_args))
                arguments.update(op_kwargs)
                q, k, v = [arguments[name] for name in ('q', 'k', 'v')]
                require(q.ndim == k.ndim == v.ndim == 4 and q.shape[0] == k.shape[0] == v.shape[0] == 1 and (q.device == k.device == v.device) and (q.dtype == k.dtype == v.dtype), 'ATTENTION_SHAPE_DTYPE')
                if not self.cpu:
                    require(q.device.type == 'cuda' and q.dtype is torch.bfloat16 and (q.shape[1:] == (19320, 12, 128)) and (k.shape[1] in (19320, 512)) and (k.shape[2:] == (12, 128)), 'FROZEN_NATIVE_B1_GEOMETRY')
                require(arguments['dropout_p'] == 0.0 and arguments['causal'] is False and (arguments['window_size_left'] == arguments['window_size_right'] == -1) and (arguments['softcap'] == 0.0) and (arguments['alibi_slopes'] is None) and (arguments['return_softmax'] is False), 'UNSUPPORTED_STOCHASTIC_OR_MASKED_ATTENTION')
                metadata = tuple(((tuple(t.shape), tuple(t.stride()), str(t.dtype), str(t.device)) for t in (q, k, v)))
                metadata += (arguments['softmax_scale'],)
                logical_bytes = q.numel() * q.element_size() + q.shape[0] * q.shape[2] * q.shape[1] * 4 + 16
                if context.is_recompute:
                    slot = state['reads']
                    require(slot < len(state['slots']) and state['slots'][slot] == (metadata, logical_bytes), 'REPLAY_ORDER_OR_METADATA_CHANGED')
                    state['reads'] += 1
                    self.record['cache_reads'] += 1
                    self.record['saved_raw_forward_calls'] += 1
                    self.live_bytes -= logical_bytes
                    require(self.live_bytes >= 0, 'NEGATIVE_LOGICAL_LIVE_CACHE')
                else:
                    require(state['reads'] == 0, 'FORWARD_AFTER_REPLAY')
                    state['slots'].append((metadata, logical_bytes))
                    state['writes'] += 1
                    self.record['cache_writes'] += 1
                    self.live_bytes += logical_bytes
                    self.record['cache_logical_peak_bytes'] = max(self.record['cache_logical_peak_bytes'], self.live_bytes)
                    self.record['all_cache_logical_peak_bytes'] = max(self.record['all_cache_logical_peak_bytes'], self.projection_live_bytes + self.live_bytes + self.ffn_live_bytes)
                    require(self.live_bytes <= self.limit, 'LOGICAL_CACHE_CAPACITY_EXCEEDED')
                    self.record['combined_cache_logical_peak_bytes'] = max(self.record['combined_cache_logical_peak_bytes'], self.live_bytes + self.ffn_live_bytes)
                    require(self.live_bytes + self.ffn_live_bytes <= self.limit + self.ffn_limit, 'COMBINED_CAPACITY_EXCEEDED')
                return checkpoint.CheckpointPolicy.MUST_SAVE
            kwargs['context_fn'] = lambda: checkpoint.create_selective_checkpoint_contexts(policy)
            return self.original(function, *args, **kwargs)
        self.installed = native_checkpoint
        checkpoint.checkpoint = native_checkpoint

        def before(opt, args, kwargs):
            require(self.pending and self.live_bytes == self.ffn_live_bytes == self.projection_live_bytes == 0, 'COMMIT_WITHOUT_COMPLETE_REPLAY')
            for state in self.pending:
                require(state['version'] == self.version and state['writes'] == state['reads'] == 2 and (state['ffn_writes'] == state['ffn_reads'] == 1) and (state['projection_writes'] == state['projection_reads'] == 2) and (state['square_counts'] == [6, 6]), 'INCOMPLETE_SELF_AND_CROSS_ATTENTION_REPLAY')

        def after(opt, args, kwargs):
            self.pending.clear()
            self.version += 1
            self.record['fake_commits'] += 1
        self.handles = [trainer.critic_optimizer.register_step_pre_hook(before), trainer.critic_optimizer.register_step_post_hook(after)]

    def receipt(self):
        require(not self.pending and self.live_bytes == self.ffn_live_bytes == self.projection_live_bytes == 0, 'LIVE_CACHE_AT_FINAL')
        return dict(self.record, live_logical_cache_bytes=self.live_bytes, live_checkpoint_records=len(self.pending), live_ffn_cache_logical_bytes=self.ffn_live_bytes, live_projection_bytes=self.projection_live_bytes, fake_version=self.version, cache_byte_scope='FA2_ABI_BYTES_ONLY_FFN_AND_COMBINED_BYTES_SEPARATE')

    def close(self):
        require(self.checkpoint.checkpoint is self.installed, 'CHECKPOINT_PATCH_OWNERSHIP')
        self.checkpoint.checkpoint = self.original
        for handle in self.handles:
            handle.remove()
