"""Semantic, model-name independent two-role gradient path.

A user declares generator/score functions and their optimizer boundaries, never a
GPU split or fastest policy. The physical pool layout/window is chosen later.
This admits a specific *gradient interface*: differentiable tensor from a
producer, scalar score objective, explicit VJP returned to producer, then one
producer optimizer commit. It is not a translator for arbitrary Python code.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Mapping
import math

from distillschedule.core.contracts import require
from distillschedule.workloads.program import Program, Call
from distillschedule.workloads.best_known.registry import fingerprint


@dataclass(frozen=True)
class RoleTask:
    """Training semantics; fields do not encode an optimization choice."""
    microbatches: int
    output_shape: tuple[int, ...]
    dtype: str = 'float32'
    score_updates: int = 0
    parameters: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = 'distillschedule.two_role_gradient.v1'

    def __post_init__(self):
        require(self.schema_version == 'distillschedule.two_role_gradient.v1',
                'unsupported role protocol', path='role.task')
        require(type(self.microbatches) is int and 1 <= self.microbatches <= 128,
                'invalid microbatch count', path='role.task')
        require(isinstance(self.output_shape, (list, tuple)) and self.output_shape and
                all(type(v) is int and v > 0 for v in self.output_shape),
                'explicit transported latent shape required', path='role.task')
        object.__setattr__(self, 'output_shape', tuple(self.output_shape))
        require(self.dtype in {'float32', 'bfloat16'}, 'unsupported gradient dtype', path='role.task')
        require(type(self.score_updates) is int and 0 <= self.score_updates <= 64,
                'invalid score update count', path='role.task')
        fingerprint(dict(self.parameters))

    @property
    def bytes_per_latent(self):
        return math.prod(self.output_shape) * (4 if self.dtype == 'float32' else 2)

    def to_dict(self):
        return {'schema_version': self.schema_version, 'microbatches': self.microbatches,
                'output_shape': list(self.output_shape), 'dtype': self.dtype,
                'score_updates': self.score_updates, 'parameters': dict(self.parameters)}

    @property
    def digest(self):
        return fingerprint(self.to_dict())

    @classmethod
    def from_dict(cls, document):
        require(isinstance(document, Mapping) and set(document)==
                {'schema_version','microbatches','output_shape','dtype','score_updates','parameters'},
                'unknown or missing role task fields', path='role.task')
        return cls(**document)

    def program(self):
        """Build true version/gradient edges without a model-name optimizer recipe."""
        calls=[]
        for index in range(self.microbatches):
            i=str(index)
            calls += [
                Call('g'+i, 'generator_forward', reads=('generator',), grad='grad',
                     attributes={'batch':index}),
                Call('f'+i, 'send_activation',inputs=('g'+i,),grad='grad'),
                Call('s'+i, 'score_vjp',inputs=('f'+i,),reads=('score',),grad='grad'),
                Call('r'+i, 'return_vjp',inputs=('s'+i,),grad='grad'),
                Call('b'+i,'generator_backward',inputs=('r'+i,'g'+i),
                     reads=('generator','gradient_g'),writes=('gradient_g',),
                     backward_of=('g'+i,),grad='grad'),
            ]
        calls.append(Call('commit_g','generator_commit',reads=('generator','optimizer_g','gradient_g'),
                          writes=('generator','optimizer_g','gradient_g'),
                          after=tuple('b'+str(i) for i in range(self.microbatches))))
        for i in range(self.score_updates):
            calls.append(Call('score_update'+str(i),'score_update',
                reads=('score','optimizer_s'),writes=('score','optimizer_s'),
                after=('commit_g',) if i == 0 else ('score_update'+str(i-1),)))
        return Program(tuple(calls),
            {'generator':0,'score':0,'optimizer_g':0,'gradient_g':0,'optimizer_s':0},
            (),(),{'task_sha256':self.digest})


@dataclass(frozen=True)
class AllocatedWorld:
    """Hardware facts. The workload factory never chooses a pool split."""
    capacities: tuple[int, ...]
    node_ids: tuple[int, ...] = ()
    host_capacities: tuple[int, ...] = ()

    def __post_init__(self):
        require(len(self.capacities)>=2 and
                all(type(v) is int and v>0 for v in self.capacities),
                'need at least two allocated devices with capacity', path='role.hardware')
        if not self.node_ids:
            object.__setattr__(self, 'node_ids', tuple(0 for _ in self.capacities))
        require(len(self.node_ids)==len(self.capacities) and
                all(type(v) is int and v>=0 for v in self.node_ids),
                'one node identity per allocated rank required', path='role.hardware')

        if not self.host_capacities:
            object.__setattr__(self, 'host_capacities', tuple(0 for _ in self.capacities))
        require(len(self.host_capacities)==len(self.capacities) and
                all(type(x) is int and x>=0 for x in self.host_capacities),
                'explicit nonnegative per-rank host budgets required',path='role.hardware')

    @property
    def world(self):return len(self.capacities)


@dataclass(frozen=True)
class PoolLayout:
    generator: tuple[int, ...]
    score: tuple[int, ...]
    window: int
    activation_policy: str = 'resident'

    def to_dict(self):
        return {'generator':list(self.generator),'score':list(self.score),
                'window':self.window,'activation_policy':self.activation_policy}

    @classmethod
    def from_dict(cls, obj):
        require(isinstance(obj,Mapping) and set(obj)=={'generator','score','window','activation_policy'},
                'invalid pool layout',path='role.layout')
        return cls(tuple(obj['generator']),tuple(obj['score']),obj['window'],obj['activation_policy'])

    def validate(self, task,hardware):
        require(self.activation_policy in {'resident','cpu_offload_prefetch'},
                'unknown activation lifetime policy',path='role.layout')
        require(self.generator and self.score and type(self.window) is int
                and 1<=self.window<=min(4,task.microbatches),
                'unsupported physical pool or window', path='role.layout')
        require(len(set(self.generator))==len(self.generator) and
                len(set(self.score))==len(self.score) and
                set(self.generator).isdisjoint(self.score) and
                set(self.generator)|set(self.score)==set(range(hardware.world)),
                'each allocated device must belong to exactly one role',path='role.layout')
        if self.activation_policy=='cpu_offload_prefetch':
            require(all(hardware.host_capacities[r]>0 for r in self.generator),
                    'offload requires declared host capacity on every producer rank',path='role.layout')
        return self

    def owners(self, index):
        return (self.generator[index%len(self.generator)],
                self.score[index%len(self.score)])



def transport_bytes_by_rank(task, layout):
    """Bound known FIFO tensor storage, NOT arbitrary model activations.

    A Generator slot retains the output, a possible contiguous send copy and
    a VJP receive buffer, regardless of saved-activation offload. Score keeps
    all send buffers until the drain and, if updating, its local input cohort.
    Two extra Score buffers cover the current input and unscaled VJP. Native
    CUDA peak accounting supplements this bound for model workspaces.
    """
    size=task.bytes_per_latent
    counts_g={r:0 for r in layout.generator};counts_s={r:0 for r in layout.score}
    for i in range(task.microbatches):
        g,s=layout.owners(i);counts_g[g]+=1;counts_s[s]+=1
    return {**{r:3*min(layout.window,n)*size for r,n in counts_g.items()},
            **{r:(n*(1+int(task.score_updates>0))+2)*size if n else 0
               for r,n in counts_s.items()}}


def enumerate_layouts(task, hardware, *, generator_bytes, score_bytes,
                      safety_bytes=0, max_candidates=128, allow_offload=False):
    """Bounded legal choices from hardware capacity, no workload-name lookup."""
    require(all(type(x) is int and x>=0 for x in (generator_bytes,score_bytes,safety_bytes)),
            'invalid role storage bounds',path='role.layout')
    require(type(max_candidates) is int and max_candidates>0,'invalid candidate bound',path='role.layout')
    require(type(allow_offload) is bool,'invalid offload capability',path='role.layout')
    feasible=[];rejected=[]
    # Two bounded physical families: contiguous rank cuts and balanced
    # per-node cuts. The latter keeps both roles on each physical node when
    # possible; it does not assume this will outperform cross-node pools.
    nodes = tuple(sorted({node for node in hardware.node_ids}))
    node_ranks = {node:tuple(r for r,n in enumerate(hardware.node_ids) if n==node)
                  for node in nodes}
    def partitions(cut):
        contiguous = tuple(range(cut))
        yield contiguous
        if len(nodes)<2 or cut<len(nodes) or hardware.world-cut<len(nodes):return
        # Allocate at least one rank per role per node and distribute the
        # remaining G ranks proportionally to available local capacity.
        per_node={node:1 for node in nodes}
        available={node:len(node_ranks[node])-2 for node in nodes}
        if any(v<0 for v in available.values()):return
        for _ in range(cut-len(nodes)):
            eligible=[node for node in nodes if available[node]>0]
            if not eligible:return
            # Balanced deterministic water filling, stable over all ranks.
            best=min(eligible,key=lambda n:(per_node[n]/len(node_ranks[n]),n))
            per_node[best]+=1;available[best]-=1
        local=tuple(sorted(r for node in nodes for r in node_ranks[node][:per_node[node]]))
        if local!=contiguous:yield local
    for cut in range(1,hardware.world):
      for generator in partitions(cut):
        score=tuple(r for r in range(hardware.world) if r not in generator)
        for window in range(1,min(4,task.microbatches)+1):
            policies=('resident','cpu_offload_prefetch') if allow_offload else ('resident',)
            for policy in policies:
                c=PoolLayout(generator,score,window,policy)
                # Model-state and known transfer bounds only. True autograd
                # workspace is observed again during native CUDA calibration.
                transport=transport_bytes_by_rank(task,c)
                if any(hardware.capacities[r]<generator_bytes+safety_bytes+transport[r]
                       for r in c.generator) or any(
                        hardware.capacities[r]<score_bytes+safety_bytes+transport[r] for r in c.score):
                    rejected.append({'layout':c.to_dict(),'reason':'ROLE_CAPACITY'});continue
                if policy=='cpu_offload_prefetch' and any(
                        hardware.host_capacities[r]<2*window*task.bytes_per_latent
                        for r in c.generator):
                    rejected.append({'layout':c.to_dict(),'reason':'HOST_CAPACITY'});continue
                feasible.append(c)
    require(feasible,'no feasible role pools under declared capacities',path='role.layout')
    # Previous prefix truncation sampled all windows of cut=1 before ever
    # considering cut=world//2, so a 16-GPU budget of four tried ONLY 1G+15S.
    # Diversity-based deterministic sampling is merely a SEARCH EXPLORATION
    # order, never an assertion that a balanced split or offload is fastest.
    # Keep exhaustive enumeration unchanged when the bound admits all choices.
    if len(feasible)<=max_candidates:
        choices=feasible
    else:
        max_window=min(4,task.microbatches)
        def local_fraction(c):
            return sum(int(any(hardware.node_ids[r]==node for r in c.generator) and
                           any(hardware.node_ids[r]==node for r in c.score))
                       for node in nodes)/len(nodes)
        def coord(c):
            return (2.0*(len(c.generator)-1)/max(1,hardware.world-2),
                    .8*(c.window-1)/max(1,max_window-1),
                    .8*int(c.activation_policy=='cpu_offload_prefetch'),
                    .9*local_fraction(c))
        def distance(a,b):
            return sum((x-y)**2 for x,y in zip(coord(a),coord(b)))
        # A legal baseline starts at the nearest feasible, roughly half/half
        # resident, window=1; if capacity forbids it, choose the closest legal.
        def baseline_key(c):
            return (c.activation_policy!='resident',
                    abs(2*len(c.generator)-hardware.world), c.window,
                    -local_fraction(c),tuple(c.generator))
        chosen=[min(feasible,key=baseline_key)]
        remaining=[c for c in feasible if c!=chosen[0]]
        while len(chosen)<max_candidates and remaining:
            # Maximize distance to the already tested family, including splits,
            # windows and activation lifecycle; stable tie-breaking is required
            # so every distributed rank enumerates the same candidates.
            next_choice=min(remaining,key=lambda c:(
                -min(distance(c,s) for s in chosen), baseline_key(c)))
            chosen.append(next_choice);remaining.remove(next_choice)
        choices=chosen
        rejected.extend({'layout':c.to_dict(),'reason':'SEARCH_BUDGET'} for c in remaining)
    return {'candidates':tuple(choices),'rejected':tuple(rejected),
            'enumeration_complete':not any(x['reason']=='SEARCH_BUDGET' for x in rejected),
            'family':'contiguous_and_node_local_two_role_pools_v2'}
