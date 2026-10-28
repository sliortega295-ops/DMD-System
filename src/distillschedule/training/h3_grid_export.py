"""Output-only bridge over W5's original-reference complete checkpoint owner."""
from copy import deepcopy
import inspect
import json
import os
from pathlib import Path
import uuid

from distillschedule.workloads.best_known.registry import file_sha256
from . import distributed
from .h3_grid_inference import require, plain

SCHEMA = 'distillschedule.h3_grid_native_output_request.v1'


def _owner(trainer, lifecycle):
    model = trainer.shared_model
    module = model.denoiser_module()
    controller = model.adaln_cache()
    return dict(clocks=deepcopy(lifecycle.state['clocks'].state_dict()),
        cursor=lifecycle.cursor.cursor, role=model._role.get(),
        modes=[(n, m.training) for n, m in module.named_modules()],
        flags=[(n, p.requires_grad, p.grad is None) for n, p in module.named_parameters()],
        cache=repr(controller.stats()))


class NativeOutputRequest:
    def __init__(self, value, output):
        self.value = plain(value)
        self.output = Path(output).expanduser().resolve()
        self.config_path = self.output.with_name(self.output.name+'.inference.json')
        self.owner_output = self.output.with_name(self.output.name+'.native-owner')
        self.result = None

    def admit(self, admitted, identity):
        from .h3_grid_inference import geometry, source_api, SOURCE_FILES
        def local():
            p = self.value
            require(isinstance(p, dict) and set(p) == {'schema_version','seed','geometry','source_terms',
                'snapshot_budget','inference_budget'} and p['schema_version'] == SCHEMA,
                'explicit complete W5 trainer.export policy required')
            require(type(p['seed']) is int and 0 <= p['seed'] < 2**63, 'independent inference seed required')
            limits = dict(max_parameters=40_000_000_000, max_export_bytes=256*2**30,
                max_table_bytes=256*2**20, host_reserve_bytes=512*2**30, device_reserve_bytes=128*2**30)
            inference = dict(max_parameters=40_000_000_000, max_loaded_bytes=512*2**30,
                max_attention_elements=100_000_000_000, max_input_elements=64_000_000, host_reserve_bytes=512*2**30)
            for b, bounds in ((p['snapshot_budget'],limits),(p['inference_budget'],inference)):
                require(isinstance(b,dict) and set(b)==set(bounds)
                    and all(type(b[k]) is int and 0 < b[k] <= v for k,v in bounds.items()), 'bounded output capacity required')
            from distillschedule.workloads.best_known.registry import repository_root
            checkout=repository_root().resolve()
            require(self.output not in (Path('/'),Path.home(),checkout) and checkout not in self.output.parents,
                    'native output must be outside the source checkout and home/filesystem roots')
            self.source = dict(root=admitted['assets']['lightx2v']['path'], files=SOURCE_FILES, terms=p['source_terms'])
            api = source_api(self.source)
            geometry(api,p['geometry'],text_dim=5120)
            require(not any(path.exists() for path in (self.output,self.config_path,self.owner_output)),
                    'fresh output, inference config and metadata-owner directory required')
            require(self.owner_output != Path(admitted['native_config']['training']['output_dir']).resolve(),
                    'output cannot write into the original training directory')
            return dict(request=p,output=str(self.output),owner_output=str(self.owner_output),identity=identity)
        row=distributed.checked(local,'W5 output request/source/capacity admission')
        require(all(other==row for other in distributed.gather(row)), 'ranks disagree on W5 output request')

    def _capacity(self, weights, table_bytes, device):
        import torch
        from .host_memory import admit_host_memory
        budget=self.value['snapshot_budget']
        count=sum(t.numel() for t in weights.values())
        size=sum(t.numel()*t.element_size() for t in weights.values())
        largest=max(t.numel()*t.element_size() for t in weights.values() if t.device.type!='cpu') if any(
            t.device.type!='cpu' for t in weights.values()) else 0
        require(count <= budget['max_parameters'] and size <= budget['max_export_bytes'],
                'native export exceeds explicit snapshot budget')
        rank=distributed.rank_world()[0]
        host=admit_host_memory(3*size if rank==0 else table_bytes,budget['host_reserve_bytes'])
        if device.type=='cuda':
            # This is a fresh output-only owner, outside every training timer.
            # Release only unused allocator blocks before a conservative driver
            # free-space check; do not count fragmented reserved bytes as free.
            torch.cuda.empty_cache()
            free,total=torch.cuda.mem_get_info(device)
            require(free >= largest+budget['device_reserve_bytes'], 'full-tensor export would breach device reserve')
            memory=dict(free_bytes=free,total_bytes=total,largest_materialized_tensor_bytes=largest,
                        reserve_bytes=budget['device_reserve_bytes'],unused_allocator_released=True,
                        timing_scope='OUTPUT_ONLY_NOT_TRAINING')
        else:
            memory={'scope':'CPU_TEST_NO_CUDA_CAPACITY_QUALIFICATION'}
        return dict(parameters=count,weight_bytes=size,host=host,device=memory)

    def _parameters(self, module, weights):
        from . import h3_grid_inference as portable
        cls=portable.model_class()
        architecture=plain({k:dict(module.config)[k] for k in inspect.signature(cls.__init__).parameters if k!='self'})
        layout={n:dict(shape=list(t.shape),dtype=str(t.dtype)) for n,t in weights.items()}
        return portable.parameters(source=self.source,architecture=architecture,geometry=self.value['geometry'],
            weight_layout=layout,seed=self.value['seed'],evidence_kind=portable.NATIVE,budget=self.value['inference_budget'])

    def publish(self, trainer, lifecycle):
        import torch
        from . import h3_grid_inference as portable
        from .checkpoint import rng_state,restore_rng
        from .inference import publish_export
        from .plugins import load_callable
        from .wan_native_inference import standalone_config
        shared=trainer.shared_model
        random=distributed.checked(lambda:rng_state(device=shared.device),'W5 pre-export rank RNG capture')
        before=distributed.checked(lambda:_owner(trainer,lifecycle),'W5 pre-export owner state')
        try:
            api=distributed.checked(lambda:portable.source_api(self.source),'W5 output original sampler source')
            module=shared.denoiser_module()
            def snapshot():
                # Reserve CPU staging BEFORE copying the selected table entries.
                from .host_memory import admit_host_memory
                budget=self.value['snapshot_budget']
                table_bytes=4*len(module.transformer_blocks)*6*6*module.config.hidden_size*2
                admit_host_memory(table_bytes,budget['host_reserve_bytes'])
                return portable.current_student_view(module,shared.adaln_cache(),api,max_table_bytes=budget['max_table_bytes'])
            weights,table=distributed.checked(snapshot,'W5 current unmerged Student/table view')
            capacity=distributed.checked(lambda:self._capacity(weights,table['table_bytes'],shared.device),'W5 pre-gather capacity')
            def projection():
                cls=portable.model_class()
                p=portable._request(self._parameters(module,weights))
                carrier,_=portable._meta(p,cls,api)
                # A separate meta model receives detached storage views. The
                # live training module, its flags and adapter selection do not.
                for name,tensor in weights.items():
                    parent,_,leaf=name.rpartition('.');owner=carrier.get_submodule(parent)
                    if leaf in owner._parameters:
                        owner._parameters[leaf]=torch.nn.Parameter(tensor.detach(),requires_grad=False)
                    else:
                        owner._buffers[leaf]=tensor.detach()
                return p,portable.ModelComponents(state={'student':carrier},metadata=portable._metadata(p)).validate()
            p,components=distributed.checked(projection,'W5 exact full converted Student projection')
            require(all(row==p for row in distributed.gather(p)), 'ranks disagree on output architecture/storage')
            _,source=distributed.checked(lambda:load_callable(portable.FACTORY),'W5 inference factory source')
            metadata=dict(components.metadata)
            result=publish_export(components,self.output,model=dict(factory=portable.FACTORY,source=source,
                parameters=p,metadata=metadata),identity=dict(model=source,model_metadata=metadata,
                training_provenance=lifecycle.identity,grid_rollout=table),cycle=lifecycle.completed,primary_only=True)
            self.result={**result,'capacity':capacity,'output_only':dict(cycles=0,forward=0,backward=0,optimizer=0,
                ema=0,samples_consumed=0,merged_lora=False)}
            distributed.checked(lambda:require(_owner(trainer,lifecycle)==before,'output changed original owner state'),
                                'W5 post-export state preservation')
            def config():
                value=standalone_config(result,self.config_path)
                value['algorithm']['factory']='distillschedule.training.h3_grid_inference:refuse_training'
                value['data']['prepared']='NEW_W5_TEXT_CONDITIONS_NOT_TRAINING.json'
                temporary=self.config_path.with_name('.'+self.config_path.name+'.tmp-'+uuid.uuid4().hex)
                try:
                    temporary.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
                    os.link(temporary,self.config_path)
                finally:
                    temporary.unlink(missing_ok=True)
            distributed.primary(config,'W5 standalone inference configuration')
            self.result['inference_config']=str(self.config_path)
            return self.result
        except BaseException:
            if self.result is not None:
                distributed.primary(lambda:(self.output/'_SUCCESS').unlink(missing_ok=True),'W5 failed export invalidation')
            raise
        finally:
            restore_rng(random)


def export_native(config, checkpoint, output):
    from .config import TrainingConfig
    from .native import _admit, _h3_grid_training
    from .h3_grid_training import run_reference, resume_metadata, training_identity, POLICY
    from distillschedule.workloads.best_known.registry import fingerprint
    require(_h3_grid_training(config,'W5') and config.execution.get('mode')=='reference',
            'only original fresh Grid reference checkpoints are supported; optimized backward gate is not bypassed')
    raw=config.to_dict()
    request=raw['trainer'].pop('export',None)
    require(isinstance(request,dict),'explicit W5 trainer.export policy required')
    raw['trainer']['resume_from']=str(Path(checkpoint).expanduser().resolve())
    training=TrainingConfig.from_dict(raw,source=config.source)
    # Ordinary admission correctly rejects terminal TRAINING restart. Its
    # source/input/config half is reused without pretending this is a restart;
    # explicit output-only checkpoint admission still runs before construction.
    admission=deepcopy(raw);admission['trainer'].pop('resume_from',None)
    admitted_config=TrainingConfig.from_dict(admission,source=config.source)
    def admit():
        result=_admit(admitted_config)
        workload,source,paths,value=result
        require(workload=='W5' and not source['identity']['dirty'],'clean original W5 source required')
        identity=training_identity(source['identity']['head'],value['assets'],value['inputs'],training.trainer['cycles'])
        policy=dict(schema_version=POLICY,identity_sha256=fingerprint(identity))
        resume_metadata(training.trainer['resume_from'],identity=identity,policy=policy,target=training.trainer['cycles'],
            torch_version=value['assets']['dependency_versions']['torch'],output_only=True)
        return result
    callback=NativeOutputRequest(request,output)
    run_reference(training,admit=admit,native_output=callback)
    require(callback.result is not None,'original owner returned without exporting restored W5 Student')
    return callback.result
