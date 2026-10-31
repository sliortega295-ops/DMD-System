"""Output-only bridge from the original W6 checkpoint owner to common weights.

No trainer, sampler or replacement ZeRO backend lives here. Native restoration
owns all training state; independent inference uses the existing portable model.
"""
import inspect
import json
import os
from pathlib import Path
import uuid

from distillschedule.workloads.best_known.registry import file_sha256
from . import distributed
from .h3_ar_inputs import H3ARInputStream

SCHEMA = 'distillschedule.h3_ar_native_output_request.v1'


def require(ok, message):
    if not ok:
        raise ValueError('native H3 export: '+message)


class OutputInputStream(H3ARInputStream):
    """Read the saved rank record only to construct the original native owner.

    The actual restored progress remains at the saved boundary, including the
    terminal boundary. No next record is staged or counted as consumed.
    """
    def __init__(self, path, *, rank, cycles, seed, completed):
        require(type(completed) is int and 1 <= completed <= cycles, 'complete saved output boundary required')
        super().__init__(path, rank=rank, cycles=cycles, seed=seed)
        self.completed = completed
        self.construction = H3ARInputStream(path, rank=rank, cycles=cycles, seed=seed, completed=completed-1)
        self.served = False

    def stage_initial(self, *args):
        return self.construction.stage_initial(*args)

    def prepare_rank_input(self, *args, **kwargs):
        require(not self.served, 'read-only construction record already served')
        result = self.construction.prepare_rank_input(*args, **kwargs)
        self.served = True
        return result

    def commit(self, cycle):
        raise ValueError('native H3 export input is read-only')

    def graph(self, task, cycle):
        raise ValueError('native H3 export input is read-only')


class NativeOutputRequest:
    def __init__(self, value, output):
        self.value = json.loads(json.dumps(value, allow_nan=False))
        self.output = Path(output).expanduser().resolve()
        self.config_path = self.output.with_name(self.output.name+'.inference.json')
        self.result = None

    def admit(self, policy):
        from distillschedule.strategies.h3_ar_planning import policy_for_node
        from .h3_cached_assets import validate_geometry
        def local():
            p = self.value
            require(isinstance(p, dict) and set(p) == {'schema_version','seed','geometry','consolidation_budget','inference_budget'}
                and p['schema_version'] == SCHEMA, 'explicit complete trainer.export policy required')
            require(type(p['seed']) is int and 0 <= p['seed'] < 2**63, 'independent inference seed required')
            validate_geometry(p['geometry'])
            limits = dict(max_parameters=40_000_000_000,max_consolidated_bytes=256*2**30,
                          host_reserve_bytes=512*2**30,device_reserve_bytes=128*2**30)
            inference = dict(max_parameters=40_000_000_000,max_loaded_bytes=512*2**30,
                max_input_elements=64_000_000,max_attention_elements=100_000_000_000,host_reserve_bytes=512*2**30)
            for budget, bounds in ((p['consolidation_budget'],limits),(p['inference_budget'],inference)):
                require(isinstance(budget,dict) and set(budget)==set(bounds)
                    and all(type(budget[k]) is int and 0<budget[k]<=v for k,v in bounds.items()), 'bounded output capacities required')
            native = policy_for_node(policy, distributed.rank_world()[0]//8)
            require(p['consolidation_budget']['device_reserve_bytes'] >= native['gpu_reserve_bytes']
                and p['consolidation_budget']['host_reserve_bytes'] >= native['host_reserve_bytes']
                and p['inference_budget']['host_reserve_bytes'] >= native['host_reserve_bytes'],
                'output must preserve the original native capacity reserves')
            require(not self.output.exists() and not self.config_path.exists(), 'new output and inference config required')
            return dict(request=p,output=str(self.output),config=str(self.config_path))
        row = distributed.checked(local, 'W6 output request/capacity policy admission')
        require(all(other==row for other in distributed.gather(row)), 'ranks disagree on W6 output request')

    def _parameters(self, session):
        from . import h3_ar_portable as portable
        from .h3_ar_inference import _layout
        from distillschedule.strategies.h3_ar_assets import validate_checkpoint_audit
        assets = session.assets
        root = Path(assets.checkpoint_root)
        audit = validate_checkpoint_audit(root, assets.checkpoint_audit, assets.checkpoint_files)
        source = assets.source_module('train_dmd2')
        config = source.model_config(root)
        defaults = {k:v.default for k,v in inspect.signature(type(session.dit).__init__).parameters.items()
                    if k not in ('self','kwargs')}
        plain = lambda v: json.loads(json.dumps(v,allow_nan=False))
        require(plain(config)==plain(session.build['architecture'])==plain(defaults)
            and audit['files']['config.json']['sha256']==file_sha256(root/'config.json'),
            'original checkpoint architecture differs from the constructed full native model')
        layout = {name:{k:v for k,v in spec.items() if k!='bytes'}
                  for name,spec in _layout(session.adapter).items() if '.adapters.fake_score.' not in name}
        return portable.parameters(architecture=plain(config),geometry=self.value['geometry'],weight_layout=layout,
            seed=self.value['seed'],evidence_kind=portable.NATIVE,budget=self.value['inference_budget'])

    def invalidate(self):
        if self.result is not None:
            distributed.primary(lambda:(self.output/'_SUCCESS').unlink(missing_ok=True),'W6 failed owner export invalidation')

    def publish(self, session, native, *, identity, capacity_check):
        from . import h3_ar_portable as portable
        from .h3_ar_inference import consolidate_current_student, owner_state
        from .checkpoint import rng_state, restore_rng
        from .inference import publish_export
        from .plugins import load_callable
        from .wan_native_inference import standalone_config
        before = owner_state(session)
        counts = (dict(session.census),dict(session.updates),dict(session.adam))
        random = rng_state(device=session.device)
        try:
            p = distributed.checked(lambda:self._parameters(session), 'W6 exact checkpoint architecture/layout')
            require(all(row==p for row in distributed.gather(p)), 'ranks disagree on native inference model')
            weights, snapshot = consolidate_current_student(session,native,budget=self.value['consolidation_budget'])
            components = distributed.checked(lambda:portable.export_components(p,weights=weights,
                model_class=type(session.dit),adapter_class=session.native.H3Adapter), 'W6 original Student export projection')
            del weights
            distributed.checked(lambda:require(owner_state(session)==before
                and counts==(dict(session.census),dict(session.updates),dict(session.adam)),
                'output changed the original training owner or call census'),'W6 output owner preservation')
            distributed.checked(capacity_check,'W6 post-gather original capacity policy before publication')
            _, source = distributed.checked(lambda:load_callable(portable.FACTORY), 'W6 portable factory source')
            metadata = dict(components.metadata)
            exported = publish_export(components,self.output,model=dict(factory=portable.FACTORY,source=source,
                parameters=p,metadata=metadata),identity=dict(model=source,model_metadata=metadata,
                training_provenance=identity,native_snapshot={k:v for k,v in snapshot.items() if k!='host_capacity'}),
                cycle=session.input_provider.completed,primary_only=True)
            def write_config():
                config = standalone_config(exported,self.config_path)
                config['algorithm']['factory'] = 'distillschedule.training.h3_ar_portable:refuse_training'
                config['data']['prepared'] = 'NEW_ORIGINAL_H3_AV_CONDITIONS_NOT_TRAINING.json'
                temporary = self.config_path.with_name('.'+self.config_path.name+'.tmp-'+uuid.uuid4().hex)
                try:
                    temporary.write_text(json.dumps(config,indent=2,sort_keys=True,allow_nan=False)+'\n')
                    os.link(temporary,self.config_path)
                finally:
                    temporary.unlink(missing_ok=True)
            try:
                distributed.primary(write_config,'W6 independent inference configuration')
            except BaseException:
                # Preserve the failed payload, but never leave it consumable as
                # a completed public export when its matching config failed.
                distributed.primary(lambda:(self.output/'_SUCCESS').unlink(missing_ok=True),'W6 failed export invalidation')
                raise
            self.result = {**exported,'inference_config':str(self.config_path),'snapshot':snapshot,
                'output_only':dict(cycles=0,forward=0,backward=0,optimizer=0,ema=0,merged_lora=False)}
            return self.result
        finally:
            restore_rng(random)


def export_native(config, checkpoint, output):
    from .config import TrainingConfig
    from .native import _admit, _h3_ar_stream
    from .h3_ar_training import run_training
    require(_h3_ar_stream(config,'W6') and config.data.get('input_kind') in ('original_h3_ar_fresh_stream.v1','original_h3_ar_epoch_stream.v1')
        and bool(config.data.get('prepared')), 'complete fresh W6 native checkpoint required; fixed replay and W5 are not substituted')
    raw = config.to_dict()
    request = raw['trainer'].pop('export',None)
    require(isinstance(request,dict), 'explicit W6 trainer.export output policy required')
    destination = Path(output).expanduser().resolve()
    # Keep original checkpoint_every: it is part of the immutable training
    # identity. Output-only control flow, not a cadence override, skips updates.
    raw['trainer'].update(resume_from=str(Path(checkpoint).expanduser().resolve()),
        output_dir=str(destination.with_name(destination.name+'.native-owner')))
    raw['trainer'].pop('stop_after_cycle',None)
    clean = TrainingConfig.from_dict(raw,source=config.source)
    callback = NativeOutputRequest(request,output)
    status = run_training(clean,admit=lambda:_admit(clean),native_output=callback)
    require(status==0 and callback.result is not None, 'actual W6 owner did not publish output')
    return callback.result
