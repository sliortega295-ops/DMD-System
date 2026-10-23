"""SHA-bound manual timing arm. Original training source and timer unchanged."""
from pathlib import Path
import argparse,hashlib,json


def main():
    p=argparse.ArgumentParser()
    for n in ('task','assets','capacity-policy','output','fa3-kernel-root','packet'):
        p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--packet-sha256',required=True)
    p.add_argument('--restore-native-barrier',type=int,choices=(0,1),required=True)
    p.add_argument('--manual-policy',choices=('uncached-rollout-fa3-varlen-v1',),required=True)
    p.add_argument('--execute',action='store_true')
    a=p.parse_args()
    assert hashlib.sha256(a.packet.read_bytes()).hexdigest()==a.packet_sha256
    packet=json.loads(a.packet.read_text())
    assert packet['schema']=='dmd.h3_ar_manual_metadata_barrier.v1' and packet['source_head']=='69a9464997de16deafd5786338f538ba7930d9a5'
    for name,ref in packet['scripts'].items():
        assert hashlib.sha256(Path(ref['path']).read_bytes()).hexdigest()==ref['sha256'],name
    if not a.execute:
        print('INTENT_ONLY_NO_GPU');return
    from native_metadata_barrier import install_on_session,annotate_final
    from distillschedule.strategies import h3_ar_session as sm
    from distillschedule.strategies import h3_ar_worker as worker
    from distillschedule.strategies.h3_ar_task import H3ARTask,H3ARImplementation
    from distillschedule.strategies.h3_ar_assets import H3ARAssets
    class Session(sm.H3ARSession):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.manual_barrier=install_on_session(self,enabled=bool(a.restore_native_barrier))

        def finish(self,cycle):
            row=super().finish(cycle)
            row['backend_effects']['manual_native_metadata_barrier']=self.manual_barrier.finish_cycle(cycle)
            return row

    sm.H3ARSession=Session
    original_write=worker.write_new_json
    binding=dict(path=str(a.packet),sha256=a.packet_sha256)
    def writer(path,row):
        row=annotate_final(row,binding,bool(a.restore_native_barrier))
        return original_write(path,row)
    worker.write_new_json=writer
    worker.run_worker(H3ARTask.from_dict(json.loads(a.task.read_text())),
        H3ARAssets.from_dict(json.loads(a.assets.read_text())),a.output,
        mode='manual',capacity_policy=json.loads(a.capacity_policy.read_text()),
        implementation=H3ARImplementation('zero3-ap'),manual_policy_id=a.manual_policy,
        manual_backend_root=str(a.fa3_kernel_root))


if __name__=='__main__':main()
