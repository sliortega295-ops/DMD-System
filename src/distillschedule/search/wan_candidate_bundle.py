"""Typed, opt-in Wan mechanism bundle; historical time is NOT a calibration.

Only the coupled 21a source mechanism is added. Existing axes remain independent
and unchanged; this is not five booleans whose arbitrary cross-product is safe.
The v1 literal graph guards are retained. Explicit v2 adds only a bounded fresh
stream lifecycle at the original geometry; it is not a new numerical claim.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Mapping

from .contracts import require
from distillschedule.workloads.best_known.registry import fingerprint, file_sha256

SCHEMA = 'distillschedule.wan_candidate_bundle.v1'
INCUMBENT_ID = 'incumbent21a-v1'
STREAM_SCHEMA = 'distillschedule.wan_candidate_bundle.v2'
STREAM_ID = 'incumbent21a-stream-v2'
SOURCE_HEAD = '21a0993db64137489e120dd6614c48a1bf1a6002'
CAPABILITIES = {
    'grad_lora_fused_epilogue': 'DISTILLSCHEDULE_WAN13B_GRAD_LORA_FUSED_EPILOGUE',
    'modulated_norm_fusion': 'DISTILLSCHEDULE_WAN13B_MODULATED_NORM_FUSION',
    'generator_rollout_cuda_graph': 'DISTILLSCHEDULE_WAN13B_GENERATOR_ROLLOUT_CUDA_GRAPH',
    'generator_nograd_lora_cast_cache': 'DISTILLSCHEDULE_WAN13B_GENERATOR_NOGRAD_LORA_CAST_CACHE',
    'critic_grad_cuda_graph': 'DISTILLSCHEDULE_WAN13B_CRITIC_GRAD_CUDA_GRAPH',
}
FORBIDDEN = ('DISTILLSCHEDULE_WAN13B_GENERATOR_ROLLOUT_MERGED_SHADOW',
             'DISTILLSCHEDULE_WAN13B_FUSED_INPUT_LORA_A',
             'DISTILLSCHEDULE_WAN13B_SCORE_CUDA_GRAPH')
# Frozen semantic assets, NOT the historic implementation/precision/time identity.
ASSETS = {
    'bootstrap_hash': '6a713476d9839cf1ea37689d63cb1cc2b405b9f9915ca5314ccdadb67b3cd60d',
    'model_identity_sha256': 'd52cf9e2d04451c018ac50c4ee1043c4ceba4014e51549bfda2ea5a123c65f54',
    'trace_bundle_hash': '654d49015921b278ab82e2fc84ea1896692da3fd38ce82fbc500cb1ebfec46d3',
    'trace_binding_hash': 'bddcf32b379ea3d0cb36237b0c21f103ea59b1635473dd8080b7182cf3c1eded',
    'negative_manifest_hash': 'ca0de6928f83ebd450418d33f8990696763d3228069ec4ac0efb2444694cf78c',
}
MODULES = ('wan13b_lora_grad_epilogue', 'wan13b_norm_policy',
    'wan13b_modulated_norm_fusion', 'wan13b_modulated_norm_kernel',
    'wan13b_fused_layernorm_kernel', 'wan13b_fused_layernorm_backward_kernel',
    'wan13b_rollout_cuda_graph', 'wan13b_critic_grad_graph',
    'wan13b_cast_graph_composition', 'wan13b_lora_cast', 'wan13b_lora_epilogue',
    'wan13b_rollout_graph_pool')


@dataclass(frozen=True)
class WanCandidateBundle:
    schema_version: str = SCHEMA
    grad_lora_fused_epilogue: bool = False
    modulated_norm_fusion: bool = False
    generator_rollout_cuda_graph: bool = False
    generator_nograd_lora_cast_cache: bool = False
    critic_grad_cuda_graph: bool = False

    def __post_init__(self):
        require(self.schema_version in (SCHEMA, STREAM_SCHEMA), 'unsupported Wan bundle schema')
        values = [getattr(self, name) for name in CAPABILITIES]
        require(all(type(v) is bool for v in values), 'bundle capabilities must be boolean')
        require(not any(values) or all(values), 'only off or the exact coupled incumbent bundle is admitted')
        require(self.schema_version != STREAM_SCHEMA or all(values), 'stream lifecycle requires the complete bundle')

    @property
    def continuous(self):
        return self.schema_version == STREAM_SCHEMA

    @property
    def enabled(self):
        return self.generator_rollout_cuda_graph

    @property
    def id(self):
        return STREAM_ID if self.continuous else INCUMBENT_ID if self.enabled else 'none'

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        require(isinstance(value, Mapping) and set(value) == {f.name for f in fields(cls)},
                'unknown/missing Wan bundle fields')
        return cls(**value)

    @classmethod
    def parse(cls, name):
        require(name in ('none', INCUMBENT_ID, STREAM_ID), 'unregistered Wan bundle')
        return cls(schema_version=STREAM_SCHEMA if name == STREAM_ID else SCHEMA,
                   **{key: name != 'none' for key in CAPABILITIES})

    def validate_backend(self, placement, chunk_tokens, lora_epilogue):
        require(not self.enabled or (placement == 'resident' and chunk_tokens == 2048 and lora_epilogue is True),
                'incumbent bundle requires resident/chunk2048/no-grad epilogue')

    def validate_task(self, task):
        if self.enabled:
            from distillschedule.strategies.wan_task import OriginalWanTask, NATIVE_HISTORY_POLICY
            if self.continuous:
                require(isinstance(task, OriginalWanTask)
                        and task.stage_schedule_policy in ('source_normal_only', NATIVE_HISTORY_POLICY)
                        and replace(task, first_step=10, stage_schedule_policy='source_normal_only') == OriginalWanTask(),
                        'stream bundle requires original geometry/precision/G1C5EMA protocol')
                return
            require(isinstance(task, OriginalWanTask) and task == OriginalWanTask(),
                    'literal graph contract supports only the exact original Wan task; no generalized task/stream')

    def validate_assets(self, identity):
        if self.enabled:
            if self.continuous:
                require(all(isinstance(identity.get(k), str) and len(identity[k]) == 64
                            and set(identity[k]) <= set('0123456789abcdef') and identity[k] != '0'*64
                            for k in ASSETS), 'stream assets require verified content hashes')
                require(all(identity[k] == ASSETS[k] for k in ('model_identity_sha256', 'negative_manifest_hash')),
                        'stream bundle still requires the audited model and negative content')
                return
            require(all(identity.get(k) == v for k, v in ASSETS.items()),
                    'incumbent graph bundle requires exact historical immutable semantic assets')

    def validate_model_files(self, bootstrap_manifest, observed_files):
        if not self.enabled:
            return
        rows = bootstrap_manifest.get('model_files')
        require(isinstance(rows, list) and rows and fingerprint(rows) == ASSETS['model_identity_sha256'],
                'bootstrap model inventory is not the exact historical content identity')
        require(all(set(row) == {'path', 'bytes', 'sha256'} and isinstance(row['path'], str)
                    and Path(row['path']).parts[:1] == ('transformer',)
                    and len(Path(row['path']).parts) == 2 for row in rows),
                'invalid bootstrap model file inventory')
        expected = {Path(row['path']).name: row['sha256'] for row in rows}
        require(len(expected) == len(rows) and observed_files == expected,
                'actual model content differs from the immutable historical model inventory')
        self.validate_versions(dict(generator=int(bootstrap_manifest['components']['generator']['optimizer_step']),
            critic=int(bootstrap_manifest['components']['critic']['optimizer_step']),
            ema=int(bootstrap_manifest['ema']['optimization_step'])))

    def validate_versions(self, versions):
        if self.continuous:
            require(set(versions) == {'generator', 'critic', 'ema'}
                    and all(type(v) is int and v >= 0 for v in versions.values())
                    and versions['critic'] == versions['ema'] == 5*versions['generator'],
                    'stream bundle requires a complete original G1/C5/EMA5 boundary')
            return
        require(not self.enabled or versions == {'generator': 2, 'critic': 10, 'ema': 10},
                'literal graph contract requires the original G2/C10/EMA10 bootstrap')

    def validate_bootstrap(self, bootstrap, task):
        """v2 initialization is not training restart: native Adam starts at zero.

        A new content-verified bootstrap must therefore be cold G0/C0/EMA0.
        The historical synthetic G2 seed remains an explicit research initializer;
        nonzero trained state must use a complete native restart checkpoint.
        """
        self.validate_task(task)
        manifest = bootstrap.manifest
        versions = dict(generator=manifest['components']['generator']['optimizer_step'],
                        critic=manifest['components']['critic']['optimizer_step'],
                        ema=manifest['ema']['optimization_step'])
        self.validate_versions(versions)
        require(versions['critic'] == task.first_step,
                'bootstrap clocks and first input cohort differ')
        require(manifest['model_identity_sha256'] == ASSETS['model_identity_sha256'],
                'stream bootstrap model identity changed')
        require(bootstrap.manifest_hash == ASSETS['bootstrap_hash'] or versions['generator'] == 0,
                'new stream bootstrap must be cold; trained Adam/EMA state requires native resume')
        require(all(manifest['components'][r]['scheduler_step'] == versions[r]
                    for r in ('generator', 'critic')), 'bootstrap scheduler clocks changed')
        require(manifest['ema']['decay'] == 0.999 and manifest['ema']['update_after_step'] == 0,
                'stream bootstrap must retain original EMA decay/update policy')

    def admit_graph(self, graph):
        if not self.enabled:
            return graph
        from distillschedule.core.graph import Dependency, ExecutionGraph
        # A legal-schedule restriction, not a workload change. Retain the
        # historical literal sample order rather than weakening census checks.
        edges = tuple(Dependency(f'r{i}', f'r{i+1}', 'incumbent_literal_rollout_order') for i in range(4))
        return ExecutionGraph(graph.operations, (*graph.dependencies, *edges), graph.metadata)

    def validate_order(self, order):
        require(not self.enabled or tuple(k for k in order if k.startswith('r')) == ('r0','r1','r2','r3','r4'),
                'literal graph census requires the original ascending rollout sequence')

    def environment(self):
        return {**{env: str(int(getattr(self, name))) for name, env in CAPABILITIES.items()},
                **{env: '0' for env in FORBIDDEN}}

    def verify_environment(self, environment):
        require(all(environment.get(k, '0') == v for k, v in self.environment().items()),
                'executed mechanism flags differ from typed Wan bundle')

    def source_identity(self):
        root = Path(__file__).resolve().parents[1] / 'runtime'
        modules = MODULES + (('wan_bundle_lifecycle',) if self.continuous else ())
        return {'historical_source_head': SOURCE_HEAD,
                'runtime_sha256': {name: file_sha256(root / (name+'.py')) for name in modules}}

    def component_identity(self):
        return {'candidate_bundle': self.to_dict(), 'bundle_sources': self.source_identity()}


OFF = WanCandidateBundle()
INCUMBENT = WanCandidateBundle.parse(INCUMBENT_ID)
STREAM_INCUMBENT = WanCandidateBundle.parse(STREAM_ID)
