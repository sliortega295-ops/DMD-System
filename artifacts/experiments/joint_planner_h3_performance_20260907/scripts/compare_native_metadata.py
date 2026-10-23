"""CPU-only correspondence, NOT tensor-value or strict gradient parity."""
import argparse,json,math,statistics
from pathlib import Path
from distillschedule.workloads.best_known.registry import write_new_json,file_sha256
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
left=a.root/'native/jointp_h3_actions_perf_20260907_r1-calibration'
right=a.root/'native/jointp_h3_actions_perf_20260907_r1-confirmation'
failures=[];losses=[];checked=0;neutral=0;rss=0;sources=[]
for rank in range(16):
    paths=[p/'execution/ranks'/f'rank_{rank:03d}.json' for p in (left,right)]
    l,r=[json.loads(p.read_text()) for p in paths]
    sources.extend(dict(path=str(p),sha256=file_sha256(p)) for p in paths)
    assert l['integration_head']==r['integration_head']=='bf59e39b8a6fe158ad15ebae07be7df71369f5fd'
    for x,y in zip(l['cycles'],r['cycles']):
        def check(value,where):
            if not value:failures.append(dict(rank=rank,cycle=x['cycle'],field=where))
        for k in ('cumulative_updates','updates','rng_before','rng_after','grid_replay_delta','finite_losses'):
            check(x[k]==y[k],k)
        check(x['census']['samples']==y['census']['samples'],'sample_identity_sequence')
        xx={v['action']:v for v in x['action_execution']['rows']}
        yy={v['action']:v for v in y['action_execution']['rows']}
        check(set(xx)==set(yy),'action_inventory')
        for action,row in xx.items():
            other=yy[action];check(row['draws']==other['draws'],action+'/draw_metadata')
            check(row['samples']==other['samples'],action+'/samples')
            if action.startswith('c'):
                check(row['rng_before']==row['rng_after'] and other['rng_before']==other['rng_after'],action+'/consumer_rng_neutral')
                neutral+=2
            else:
                check(row['rng_before']==other['rng_before'] and row['rng_after']==other['rng_after'],action+'/rng_pair')
                checked+=1
        for role in ('student','fake'):
            for i,(v,w) in enumerate(zip(x['losses'][role],y['losses'][role])):
                assert math.isfinite(v) and math.isfinite(w)
                losses.append(dict(rank=rank,cycle=x['cycle'],role=role,index=i,calibration=v,confirmation=w,
                                   absolute_error=abs(v-w),relative_error=abs(v-w)/max(abs(v),abs(w),1e-12)))
        for k in ('memory_before','memory_after'):rss=max(rss,y[k]['process'].get('VmHWM',0))
def stats(values):
    values=sorted(values)
    return dict(count=len(values),mean=statistics.mean(values),median=statistics.median(values),
                p95=values[min(len(values)-1,math.ceil(.95*len(values))-1)],max=max(values))
out=a.root/'analysis-comparison-r1';out.mkdir(exist_ok=False)
value=dict(status='PASS_METADATA_RNG_CENSUS_CORRESPONDENCE' if not failures else 'FAIL',
    failures=failures,ranks=16,cycles=2,matched_stochastic_actions=checked,checked_rng_neutral_consumers=neutral,
    stochastic_tensor_values='NOT_RECHECKED_METADATA_ONLY_PRIOR_FULL_DRAW_PROOF',
    numerical_status='REVIEW_NOT_STRICT_PASS',loss_relative_error=stats([r['relative_error'] for r in losses]),
    loss_absolute_error=stats([r['absolute_error'] for r in losses]),
    confirmation_max_observed_rank_rss_hwm_bytes=rss,
    comparison_scope='CALIBRATION_VS_NATIVE_WINDOW3_VS_WINDOW5_NOT_A_TIMING_PAIR',sources=sources)
write_new_json(out/'summary.json',value);write_new_json(out/'loss-observations.json',losses)
write_new_json(out/'artifact_sha256.json',{p.name:file_sha256(p) for p in out.iterdir() if p.is_file()})
print(json.dumps({k:v for k,v in value.items() if k!='sources'},indent=2))
assert not failures,failures
