"""Unknown Conv3D role task; no optimization recipe, placement or search import."""
from __future__ import annotations
import torch
from torch import nn
from distillschedule.workloads.best_known.registry import fingerprint
from distillschedule.open_workload.role_program import RoleTask
from distillschedule.open_workload.role_transport import RoleFunctions


def make_model(role,device):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(101 if role=='generator' else 202)
        return nn.Sequential(nn.Conv3d(2,4,1),nn.SiLU(),nn.Conv3d(4,2,1)).to(device)


def make_optimizer(role,module):
    return torch.optim.Adam(module.parameters(),lr=1e-3)


def sample(index,role,device):
    params=PARAMETERS
    shape=(1,2,params['frames'],params['height'],params['width'])
    # Explicit deterministic data/conditioning. Real workloads bind actual
    # dataset tensor sources here, not optimization suggestions.
    return torch.full(shape,.1*(index+1) if role=='generator' else .03*(index+2),device=device)


def generator_forward(module,latent):return module(latent)


def score_loss(module,latent,condition):return (module(latent)-condition).square().mean()


def score_train_loss(module,latent,condition):return (module(latent.detach())-condition).square().mean()

PARAMETERS={}

def build(parameters):
    global PARAMETERS
    PARAMETERS=dict(parameters)
    task=RoleTask(parameters['microbatches'],
        (1,2,parameters['frames'],parameters['height'],parameters['width']),
        score_updates=parameters.get('score_updates',0),
        parameters=dict(parameters))
    functions=RoleFunctions(make_model,make_optimizer,sample,generator_forward,score_loss,
        score_train_loss if task.score_updates else None,
        data_identity=fingerprint({'data':'synthetic_declared_C3D_example','parameters':parameters}),
        role_storage_estimate_bytes={'generator':20000,'score':20000})
    return task,functions
