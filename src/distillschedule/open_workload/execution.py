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
        self.native_owner=False
        self.key='torch-callable-replicated-v1' if self.distributed else 'torch-callable-resident-v1'
        self.device=None if self.distributed else str(workload.device)
        if self.distributed:
            from .topology import discover
            self.topology = discover()
            from .replicated import checked
            self.device=checked(lambda: workload.device.type,'execution device')
            from .collective_owner import present, contract, BACKEND
            flags=self.consensus(checked(lambda: present(workload),'native owner capability'))
            require(all(flag==flags[0] for flag in flags),
                    'rank native owner capabilities differ',path='open.backend')
            self.native_owner=flags[0]
            if self.native_owner:
                self.native_binding=contract(workload)
                self.key=BACKEND
            else:
                checked(lambda: require(workload.replica_semantics is not None,
                                        'distributed execution requires replica semantics',path='open.backend'), 'semantics')
        else:
            from .collective_owner import present
            require(not present(workload),
                    'native collectives require an explicitly initialized multi-rank group',path='open.backend')

    def consensus(self,value):
        if not self.distributed: return [value]
        from .replicated import gather
        return gather(value)

    def identity(self):
        if self.native_owner:
            from .collective_owner import identity
            return identity(self.workload)
        if not self.distributed:
            # Calibration/plan artifacts use JSON, including algorithm-native
            # tuples. Compare that same complete canonical identity locally,
            # just as below for replicas; never drop source/state/policy fields.
            return json.loads(json.dumps(self.workload.identity, allow_nan=False))
        from .replicated import identity
        rows=identity(self.workload)
        first=rows['ranks'][0]
        return json.loads(json.dumps({**first['common'], 'device':self.device,
                'feeds':fingerprint([r['feeds'] for r in rows['ranks']]), 'replicas':rows}))

    def hardware(self):
        from .planning import hardware_identity
        value=hardware_identity(self.workload.device)
        if not self.distributed: return value
        return {'replicas':self.consensus(value),'world_size':self.world,'topology':self.topology}

    def inventory(self,limit):
        if self.native_owner:
            from .collective_owner import inventory
            return inventory(self.workload,limit=limit)
        if not self.distributed:
            from .torch_backend import candidates
            return candidates(self.workload,limit=limit)
        from .replicated import inventory,checked
        result=checked(lambda: inventory(self.workload,limit=limit,topology=self.topology),'capability matching')
        rows=self.consensus(result)
        require(all(r==result for r in rows),'rank candidate inventories differ',path='open.backend')
        return result

    def session(self,implementation, *, copy_state=True, persistent=False):
        if self.native_owner:
            from .collective_owner import make_session
            return make_session(self.workload,implementation,copy_state=copy_state,
                                persistent=persistent,binding=self.native_binding)
        if self.distributed:
            from .replicated import make_session
            return make_session(self.workload,implementation,copy_state=copy_state,persistent=persistent)
        from .torch_backend import Session
        return Session(self.workload,implementation,copy_state=copy_state,persistent=persistent)

    def resident_trial(self, budget):
        """Private CPU/single-CUDA budgeted opt-in; public clone policy is unchanged."""
        require(not self.native_owner,
                'native collective owner resident trials/calibration are not admitted',path='open.backend')
        from .resident_trial import ResidentTrial
        return ResidentTrial(self, budget)

    def graph(self, implementation):
        if self.native_owner:
            from .collective_owner import implementation as admit
            admit(implementation)
            return self.workload.program.compile()
        if not self.distributed: return self.workload.program.compile()
        from .replica_contract import execution_graph
        reduction=implementation['reduction']
        if reduction == 'per_module':
            from .mixed_layout import graph
            return graph(self.workload,implementation)
        return execution_graph(self.workload.program, self.workload.replica_semantics,
                               'fsdp_forward' if reduction.startswith(('fsdp2_', 'hsdp2_')) else reduction)

    def compare(self,expected,actual,**kwargs):
        from .torch_backend import compare
        result=compare(expected,actual,**kwargs)
        if not self.distributed: return result
        rows=self.consensus(result)
        return {'status':'PASS' if all(r['status']=='PASS' for r in rows) else 'REVIEW',
                'rank_comparisons':rows,**kwargs}

    def observations(self,session):
        row={'profile':session.profile,'static_bytes':session.static_bytes,'effects':session.effects}
        if self.native_owner:
            # Keep the local lifecycle counters usable on the next bound cycle;
            # do not recursively aggregate an earlier rank-effects collection.
            row['effects']={key:value for key,value in session.effects.items()
                            if key!='native_collective_owner_effects'}
        if not self.distributed:
            self.last_observations=[row];return
        # DDP owns persistent flat bucket storage not exposed by Module.parameters.
        # This conservative accounting term is recorded, not silently treated as 0.
        extra=0
        if self.native_owner:
            extra=session.lifecycle.runtime['bucket_storage_bound_bytes']
            # No generic reducer exists on this path. Use only the actual
            # source-bound owner's validated runtime buffer accounting.
            row['static_bytes']-=getattr(session,'_native_observation_extra',0)
        elif session.lifecycle.option['reduction']=='ddp_overlap':
            extra=2*sum(p.numel()*p.element_size() for key in session.lifecycle.module_keys
                        for p in session.state[key].parameters() if p.requires_grad)
        if not self.native_owner and session.lifecycle.option['reduction']=='per_module':
            extra=session.lifecycle.bucket_storage_bound_bytes
        row['bucket_storage_bound_bytes']=extra
        if not self.native_owner and (session.lifecycle.option['reduction'].startswith(('fsdp2_', 'hsdp2_')) or session.lifecycle.option['reduction']=='per_module'):
            # Tensor accounting cannot see all flattened all-gather workspaces on
            # CPU. Include an explicit conservative staging allowance; not RSS.
            staging=2*session.lifecycle.full_parameter_bytes
            row['fsdp_staging_allowance_bytes']=staging
            row['profile']={key:{**value,'workspace_bytes':value['workspace_bytes']+staging}
                            for key,value in row['profile'].items()}
        row['static_bytes']+=extra
        rows=self.consensus(row)
        require(all(set(r['profile'])==set(row['profile']) for r in rows),
                'rank operation profiles differ',path='open.backend')
        self.last_observations=rows
        session.profile={key:{metric:max(r['profile'][key][metric] for r in rows)
                         for metric in ('duration_ns','retained_bytes','workspace_bytes')}
                         for key in row['profile']}
        session.static_bytes=max(r['static_bytes'] for r in rows)
        if self.native_owner:
            session._native_observation_extra=extra
            session.effects={**row['effects'],'native_collective_owner_effects':[r['effects'] for r in rows]}
        else:
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
        from .replicated import checked, broadcast
        value=checked(lambda:fn() if self.rank==0 else None,'planning')
        return broadcast(value)
