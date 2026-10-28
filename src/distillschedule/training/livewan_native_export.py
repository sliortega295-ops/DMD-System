"""Read-only output seam over the actual dual-pool full-C51 checkpoint owner."""
from copy import deepcopy
import json
import os
from pathlib import Path
import uuid

from . import distributed
from .livewan_native_inference import require,plain

SCHEMA='distillschedule.livewan_native_output_request.v1'


def _owner(lifecycle,bundle):
    modules=[getattr(bundle,k) for k in ('student','critic','critic_shadow','teacher') if getattr(bundle,k) is not None]
    return dict(progress=deepcopy(lifecycle.progress.state_dict()),pending=deepcopy(lifecycle.loader.pending),
        modes=[[(n,m.training) for n,m in module.named_modules()] for module in modules],
        flags=[[(n,p.requires_grad,p.grad is None) for n,p in module.named_parameters()] for module in modules])


class NativeOutputRequest:
    def __init__(self,value,output):
        self.value=plain(value);self.output=Path(output).expanduser().resolve()
        self.owner_output=self.output.with_name(self.output.name+'.native-owner')
        self.config_path=self.output.with_name(self.output.name+'.inference.json')
        self.codec,self.result=None,None

    def admit(self,admitted,identity):
        from distillschedule.workloads.best_known.registry import repository_root
        from .livewan_native_inference import budget,geometry,NATIVE
        def local():
            p=self.value
            require(type(p) is dict and set(p)=={'schema_version','seed','geometry','budget','codec'}
                and p['schema_version']==SCHEMA,'complete explicit C51 trainer.export policy required')
            require(type(p['seed']) is int and 0 <= p['seed'] < 2**63,'independent inference seed required')
            budget(p['budget']);geometry(p['geometry'],NATIVE)
            checkout=repository_root().resolve()
            require(self.output not in (Path('/'),Path.home(),checkout) and checkout not in self.output.parents,
                    'fresh project output outside source and home/filesystem roots required')
            require(not any(p.exists() for p in (self.output,self.owner_output,self.config_path)),
                    'fresh output, inference config and owner metadata directory required')
            codec=p['codec'];pins=codec.get('weights') if isinstance(codec,dict) else None
            require(isinstance(pins,list) and pins and all(type(v) is dict and type(v.get('bytes')) is int
                and v['bytes']>0 for v in pins),'explicit pinned original codec shards required')
            return dict(request=p,output=str(self.output),identity=identity)
        row=distributed.checked(local,'C51 output request before native model construction')
        require(all(other==row for other in distributed.gather(row)),'ranks disagree on C51 output request')
        def codec():
            from .host_memory import admit_host_memory
            from .helios_public_assets import REAL
            from .wan_native_inference import _load_codec
            size=sum(v['bytes'] for v in self.value['codec']['weights'])
            require(3*size<=self.value['budget']['max_loaded_bytes'],'original codec exceeds capacity')
            admit_host_memory(3*size,self.value['budget']['host_reserve_bytes'])
            self.codec,_=_load_codec(self.value['codec'],self.value['budget'],REAL)
        distributed.primary(codec,'C51 primary-only original codec admission')

    def _parameters(self,student):
        from . import livewan_native_inference as consumer
        from .wan_native_inference import codec_parameters
        from .helios_public_assets import REAL
        return plain(dict(schema_version=consumer.SCHEMA,source=consumer.inference_source(),
            runtime=consumer.runtime_identity(),architecture=consumer.architecture(student),
            geometry=self.value['geometry'],precision=consumer.PRECISION,
            seed=self.value['seed'],evidence_kind=consumer.NATIVE,budget=self.value['budget'],
            codec=codec_parameters(self.codec,evidence_kind=REAL,provenance=self.value['codec']['provenance'])))

    def _project(self,student):
        import torch
        from torch.distributed.tensor import DTensor
        from . import livewan_native_inference as consumer
        from .adapters import ModelComponents
        from .host_memory import admit_host_memory
        require(student is not None,'primary rank does not own the actual current Student')
        parameters=self._parameters(student)
        p,probe,codec,sizes=consumer._probe(parameters,constructor=type(student))
        require(all(not getattr(m,'merged',False) for m in student.modules()),'merged Student is forbidden')
        for actual,expected,label in ((student,probe,'Student'),(self.codec,codec,'codec')):
            a,b=actual.state_dict(),expected.state_dict()
            require(a.keys()==b.keys() and all(not isinstance(a[k],DTensor) and a[k].shape==b[k].shape
                and a[k].dtype==b[k].dtype==torch.float32 and a[k].device.type!='meta' for k in b),
                'complete exact FP32 replica state/layout required: '+label)
        require(student.freqs.shape==probe.freqs.shape and student.freqs.dtype==probe.freqs.dtype
            and student.freqs.device.type!='meta','actual original RoPE layout/precision required')
        capacity=admit_host_memory(3*sizes['state_bytes'],p['budget']['host_reserve_bytes'])
        components=ModelComponents(state=dict(generator=consumer.current_view(student),codec=self.codec),
                                   metadata=consumer.metadata(p)).validate()
        return components,p,capacity

    def publish(self,lifecycle,bundle,*,rank):
        from .checkpoint import rng_state,restore_rng
        from .inference import publish_export
        from .plugins import load_callable
        from . import livewan_native_inference as consumer
        from .livewan_c51_state import physical_versions
        # These read-only checks run on ALL physical roles before publication.
        # No next row, C shadow broadcast, rollout or optimizer call is entered.
        def boundary():
            require(lifecycle.header is not None and lifecycle.loader.cycle==lifecycle.header['cycle']
                and lifecycle.progress.actual==physical_versions(rank,lifecycle.header['cycle'])
                and (bundle.student is not None)==(rank<8),'current restored physical role/cycle differs')
            for state in lifecycle.state.values():
                getattr(state,'validate_checkpoint_boundary',lambda:None)()
            return _owner(lifecycle,bundle)
        before=distributed.checked(boundary,'C51 complete restored output boundary')
        random=distributed.checked(rng_state,'C51 output rank RNG snapshot')
        published=False
        try:
            projected=distributed.primary(lambda:self._project(bundle.student),'C51 complete primary G projection')
            parameters,capacity=distributed.gather((projected[1],projected[2]) if rank==0 else None)[0]
            _,source=distributed.checked(lambda:load_callable(consumer.FACTORY),'C51 portable consumer identity')
            metadata=consumer.metadata(parameters)
            exported=publish_export(projected[0] if rank==0 else None,self.output,
                model=dict(factory=consumer.FACTORY,source=source,parameters=parameters,metadata=metadata),
                identity=dict(model=source,model_metadata=metadata,training_provenance=lifecycle.identity),
                cycle=lifecycle.header['cycle'],primary_replica=True)
            published=True
            def unchanged():
                require(_owner(lifecycle,bundle)==before,'output changed native modes/flags/role/progress')
                lifecycle.capacity_check()
            distributed.checked(unchanged,'C51 output preserves complete native owner')
            def config():
                from .wan_native_inference import standalone_config
                raw=standalone_config(exported,self.config_path)
                raw['algorithm']['factory']=consumer.__name__+':refuse_training'
                raw['data']['prepared']='NEW_CLEAN_WORLD_AND_POSITIVE_TEXT.json'
                temporary=self.config_path.with_name('.'+self.config_path.name+'.tmp-'+uuid.uuid4().hex)
                try:
                    temporary.write_text(json.dumps(raw,indent=2,sort_keys=True)+'\n')
                    os.link(temporary,self.config_path)
                finally:
                    temporary.unlink(missing_ok=True)
            distributed.primary(config,'C51 standalone inference configuration')
            self.result={**exported,'inference_config':str(self.config_path),'capacity':capacity,
                'output_transport':'COMPLETE_PRIMARY_RESIDENT_REPLICA_NO_SCORE_WEIGHT_COPY',
                'output_only':dict(cycles=0,forward=0,backward=0,optimizer=0,ema=0,samples_consumed=0,merged_lora=False)}
            def receipt():
                from distillschedule.runtime.livewan_crossscale_worker import atomic_json
                from distillschedule.workloads.best_known.registry import file_sha256
                atomic_json(lifecycle.output/'continuation'/f'rank-{rank:05d}.output.json',dict(
                    schema_version=SCHEMA,status='EXPORTED_COMPLETE_RESTORED_STATE_READ_ONLY',
                    identity=lifecycle.identity,rank=rank,cycle=lifecycle.header['cycle'],
                    physical_versions=dict(lifecycle.progress.actual),progress=lifecycle.progress.state_dict(),
                    export_manifest_sha256=file_sha256(self.output/'export.json'),
                    output_only=self.result['output_only'],quality_validation='NOT_RUN'))
            distributed.checked(receipt,'C51 all-rank output-only state receipt publication')
            return self.result
        except BaseException:
            if published:
                distributed.primary(lambda:(self.output/'_SUCCESS').unlink(missing_ok=True),
                                    'C51 invalidation after output-only preservation/config failure')
            raise
        finally:
            distributed.checked(lambda:restore_rng(random),'C51 output RNG restoration')


def export_native(config,checkpoint,output):
    from .config import TrainingConfig
    from .native import _admit,_livewan_c51_stream
    from .livewan_c51_training import run_training
    require(_livewan_c51_stream(config,'W4'),'full fresh C51 owner required; no sequential/reference substitution')
    raw=config.to_dict();request=raw['trainer'].pop('export',None)
    require(isinstance(request,dict),'explicit C51 trainer.export policy required')
    callback=NativeOutputRequest(request,output)
    raw['trainer']['resume_from']=str(Path(checkpoint).expanduser().resolve())
    raw['trainer']['output_dir']=str(callback.owner_output)
    # Cadence/horizon/seed are identity-bearing; never set checkpoint_every=0.
    clean=TrainingConfig.from_dict(raw,source=config.source)
    def admit():
        result=_admit(clean)
        require(not result[1]['identity']['dirty'],'committed original checkpoint source required')
        return result
    status=run_training(clean,admit=admit,native_output=callback)
    require(status==0 and callback.result is not None,'native C51 owner did not publish current G')
    return callback.result
