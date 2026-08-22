import argparse,json,math
from pathlib import Path
import torch
from safetensors.torch import load_file

def pair(paths_a,paths_b):
    ss=rr=cc=dot=0.0; max_abs=0.0; n=0
    for pa,pb in zip(paths_a,paths_b):
        a=load_file(str(pa),device='cpu'); b=load_file(str(pb),device='cpu')
        assert set(a)==set(b)
        for k in a:
            x=a[k].float().double(); y=b[k].float().double(); d=y-x
            ss += float((d*d).sum()); rr += float((x*x).sum()); cc += float((y*y).sum()); dot += float((x*y).sum());
            max_abs=max(max_abs,float(d.abs().max())); n+=x.numel()
    return {'numel':n,'normalized_l2':math.sqrt(ss/max(rr,1e-300)),'cosine':dot/math.sqrt(max(rr*cc,1e-300)),'max_abs':max_abs}

def paths(summary):
    x=json.load(open(summary)); e=x['correctness_evidence'];
    return e

def compare(a,b):
    A=paths(a);B=paths(b); out={}
    out['rollout']=pair([Path(x['path']) for x in A['rollouts']],[Path(x['path']) for x in B['rollouts']])
    for c in ('generator','critic','generator_optimizer','critic_optimizer'):
        out[c]=pair([Path(A['final_trainables'][c]['path'])],[Path(B['final_trainables'][c]['path'])])
    return out
p=argparse.ArgumentParser();p.add_argument('--a');p.add_argument('--b');p.add_argument('--output');q=p.parse_args();o=compare(q.a,q.b);Path(q.output).write_text(json.dumps(o,indent=2,sort_keys=True)+'\n');print(json.dumps(o,indent=2,sort_keys=True))
