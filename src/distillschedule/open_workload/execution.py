"""One planning workflow, topology-selected executable backend.

Selection is by initialized resources and semantic capabilities, never model
names. A group must be explicitly launched/initialized by the caller. The local
and replicated providers share calibration, cost records and the DP service.
"""
from __future__ import annotations
from pathlib import Path
import json
from distillschedule.core.contracts import require
from distillschedule.workloads.best_known.registry import fingerprint, write_new_json


class ExecutionBackend:
    def __init__(self, workload):
        import torch.distributed as dist
        self.workload=workload
        self.distributed=dist.is_initialized() and dist.get_world_size()>1
        self.world=dist.get_world_size() if self.distributed else 1
        self.rank=dist.get_rank() if self.distributed else 0
        self.key='torch-callable-replicated-v1' if self.distributed else 'torch-callable-resident-v1'
        self.device=workload.device.type if self.distributed else str(workload.device)
        if self.distributed:
            from .replicated import checked
            checked(lambda: require(workload.replica_semantics is not None,
                                    'distributed execution requires replica semantics',path='open.backend'), 'semantics')

    def consensus(self,value):
        if not self.distributed: return [value]
        from .replicated import gather
        return gather(value)

    def identity(self):
        if not self.distributed: return self.workload.identity
        from .replicated import identity
        rows=identity(self.workload)
        first=rows['ranks'][0]
        return json.loads(json.dumps({**first['common'], 'device':self.device,
                'feeds':fingerprint([r['feeds'] for r in rows['ranks']]), 'replicas':rows}))

    def hardware(self):
        from .planning import hardware_identity
        value=hardware_identity(self.workload.device)
        if not self.distributed: return value
        return {'replicas':self.consensus(value),'world_size':self.world}

    def inventory(self,limit):
        if not self.distributed:
            from .torch_backend import candidates
            return candidates(self.workload,limit=limit)
        from .replicated import inventory,checked
        result=checked(lambda: inventory(self.workload,limit=limit),'capability matching')
        rows=self.consensus(result)
        require(all(r==result for r in rows),'rank candidate inventories differ',path='open.backend')
        return result

    def session(self,implementation):
        if self.distributed:
            from .replicated import make_session
            return make_session(self.workload,implementation)
        from .torch_backend import Session
        return Session(self.workload,implementation)

    def compare(self,expected,actual,**kwargs):
        from .torch_backend import compare
        result=compare(expected,actual,**kwargs)
        if not self.distributed: return result
        rows=self.consensus(result)
        return {'status':'PASS' if all(r['status']=='PASS' for r in rows) else 'REVIEW',
                'rank_comparisons':rows,**kwargs}

    def observations(self,session):
        row={'profile':session.profile,'static_bytes':session.static_bytes,'effects':session.effects}
        if not self.distributed:
            self.last_observations=[row];return
        # DDP owns persistent flat bucket storage not exposed by Module.parameters.
        # This conservative accounting term is recorded, not silently treated as 0.
        extra=0
        if session.lifecycle.option['reduction']=='ddp_overlap':
            extra=2*sum(p.numel()*p.element_size() for key in session.lifecycle.module_keys
                        for p in session.state[key].parameters() if p.requires_grad)
        row['bucket_storage_bound_bytes']=extra
        row['static_bytes']+=extra
        rows=self.consensus(row)
        require(all(set(r['profile'])==set(row['profile']) for r in rows),
                'rank operation profiles differ',path='open.backend')
        self.last_observations=rows
        session.profile={key:{metric:max(r['profile'][key][metric] for r in rows)
                         for metric in ('duration_ns','retained_bytes','workspace_bytes')}
                         for key in row['profile']}
        session.static_bytes=max(r['static_bytes'] for r in rows)
        session.effects={'replica_effects':[r['effects'] for r in rows]}

    def validated(self,fn,label):
        if not self.distributed: return fn()
        from .replicated import checked
        return checked(fn,label)

    def primary(self,fn,label):
        if not self.distributed: return fn()
        from .replicated import checked
        return checked(lambda: fn() if self.rank==0 else None,label)

    def mkdir(self,path):
        return self.primary(lambda: Path(path).mkdir(parents=True,exist_ok=False),'output creation')

    def publish(self,path,value):
        # All ranks must consume one shared artifact, not their local-clock winner.
        if self.distributed:
            hashes=self.consensus(fingerprint(value))
            require(len(set(hashes))==1,'rank publication contents differ',path='open.backend')
        return self.primary(lambda: write_new_json(path,value),'artifact publication')

    def select_plan(self,fn):
        if not self.distributed: return fn()
        from .replicated import checked
        import torch.distributed as dist
        value=checked(lambda:fn() if self.rank==0 else None,'planning')
        box=[value];dist.broadcast_object_list(box,src=0)
        return box[0]
