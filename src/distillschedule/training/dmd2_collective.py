"""Opt-in FP32 official DMD2 role replicas in the public training lifecycle.

This is a distributed correctness path, not promotion of the separately
measured mixed-precision/FSDP2/fused-Adam portfolio. No benchmark launcher,
fixed cycle count, trace path or second training loop is used here.
"""
from copy import deepcopy

import torch.distributed as dist

from .dmd2_owner import DMD2Owner
from .sd import require
from .native_roles import RoleReplicas
from .single_device import admit_fp32


class DMD2CollectiveOwner(DMD2Owner):
    def __init__(self, model, args, source, uncond_embedding, *, identity, seed):
        self._collective = deepcopy(identity['native_collective'])
        self._role_replicas = None
        super().__init__(model, args, source, uncond_embedding, identity=identity, seed=seed)
        self._role_replicas = RoleReplicas(model,
            dict(generator='feedforward_model', guidance='guidance_model'), self.device)

    def _validate_process_ownership(self):
        device = admit_fp32({'device': str(self.device)}, require=require, allow_replicated=True)
        require(dist.is_initialized() and dist.get_world_size() > 1
                and device == self.device and dist.get_world_size() == self._collective['world_size']
                and self._collective == self.identity['native_collective'],
                'native role source/topology/precision ownership changed')
        if self._role_replicas is not None and self._role_replicas.wrappers:
            self._role_replicas.validate()

    def _identity_device(self):
        # cuda:N is rank-local ownership; the shared training identity is the
        # device family plus explicit topology. Actual local index is checked.
        return self.device.type

    def __deepcopy__(self, memo):
        raise ValueError('native collective owners cannot be cloned or used in paired trials')

    def _scheduler_clock(self, step):
        return step * self._collective['world_size']

    def _step_scheduler(self, role):
        # Accelerate's original scheduler advances world times per iteration.
        # Both clocks advance even on non-G native iterations; Adam does not.
        for _ in range(self._collective['world_size']):
            self.schedulers[role].step()

    def admit_execution(self, mode, implementation):
        from distillschedule.open_workload.collective_owner import implementation as admit
        require(mode in ('reference', 'manual'),
                'native collective auto/plan calibration is not qualified')
        if mode == 'reference':
            require(not implementation, 'reference native role backend cannot override its implementation')
        else:
            admit(implementation)

    def execution_options(self):
        self._validate_live()
        # No singleton clone-based owner override may enter this persistent
        # distributed owner. Its actual role backend is installed separately.
        baseline = {'gradient_checkpointing': self.args.gradient_checkpointing}
        return dict(schema_version='distillschedule.owner_execution.v1', baseline=baseline,
            options=[baseline], effect=dict(kind='native_checkpoint', enabled_key='gradient_checkpointing',
                                           components=['generator', 'fake']),
            identity=dict(source=self.source.identity, native_collective=deepcopy(self._collective),
                scope='FP32_NATIVE_ROLE_DDP_NO_LOCAL_OVERRIDES',
                capability=dict(status='UNAVAILABLE_NATIVE_COLLECTIVE_OWNER_OVERRIDES')))

    def collective_execution_spec(self):
        self._validate_live()
        return dict(schema_version='distillschedule.native_collective_owner.v1', backend='ddp',
            world_size=self._collective['world_size'], device_type=self.device.type,
            shared_identity=dict(source=self.source.identity, identity=self.identity,
                                 role_paths=self._role_replicas.paths))

    def prepare_collective_execution(self):
        self._validate_live()
        from .distributed import gather
        # The original Trainer seeds construction with seed+rank, including
        # its unused frozen DummyNetwork. Native DDP then broadcasts complete
        # role trees from rank0. Qualify that exact initialization transition:
        # real pretrained frozen tensors must already agree; ONLY the native
        # randomly initialized dummy may acquire rank0's frozen baseline.
        rows = gather(self._frozen_identity)
        require(all(row.keys() == rows[0].keys() for row in rows),
                'native frozen initialization layout differs across ranks')
        for name in rows[0]:
            if not name.startswith('guidance_model.dummy_network.'):
                require(all(row[name] == rows[0][name] for row in rows),
                        'pretrained frozen role initialization differs: ' + name)
        expected = {name: rows[0][name] for name in self._frozen_identity}
        self._role_replicas.prepare()
        require(self._frozen_tensors_identity() == expected,
                'native DDP did not preserve/broadcast the qualified frozen initialization')
        self._frozen_identity = deepcopy(expected)
        self.frozen_hashes = {name: expected[name]['sha256'] for name in self.frozen_hashes}
        self._validate_live()

    def validate_collective_execution(self):
        self._role_replicas.validate()
        self._validate_live()
        counts = deepcopy(self._role_replicas.counters)
        counts['generator']['optimizer_commits'] = self.generator_updates
        counts['guidance']['optimizer_commits'] = self.guidance_updates
        trainable = {name for group in self.ownership.values() for names in group for name in names}
        # Conservative persistent DDP bucket + grad-view allowance. Not an
        # allocator observation or a claim that unused memory is zero.
        buckets = 2 * sum(p.numel() * p.element_size() for name, p in self._model_named('parameters')
                          if name in trainable)
        return dict(world_size=dist.get_world_size(), rank=dist.get_rank(), roles=counts,
            zero_gradients=True, bucket_storage_bound_bytes=buckets,
            forward_counter_scope='CURRENT_PROCESS_WRAPPER_INSTALL',
            optimizer_counter_scope='PERSISTENT_NATIVE_TRAINING_CLOCK',
            precision='FP32', numerical_status='NOT_RUN', gpu='NOT_RUN')

    def validate_checkpoint_progress(self, value, progress):
        cycle = value['native_step'] // self.ratio
        require(progress == dict(cycle=cycle, cursor=cycle * self._collective['world_size'],
                                 versions={'native_cycle': cycle}),
                'inner native clocks and rank-strided Program progress disagree')
