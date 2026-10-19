"""Explicit SHA-bound source addition, native worker, no planner/default change."""
from pathlib import Path
import argparse,hashlib,json,subprocess,sys
from stream_execution import explicit_worker

def main():
    p=argparse.ArgumentParser();p.add_argument('--stream-packet',required=True);p.add_argument('--stream-packet-sha256',required=True)
    a,rest=p.parse_known_args();path=Path(a.stream_packet)
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    assert sha(path)==a.stream_packet_sha256
    packet=json.loads(path.read_text())
    assert set(packet)=={'schema_version','source_head','arm','run_id','entry_files','backend_packet','backend_packet_sha256'}
    assert packet['schema_version']=='dmd.wan_post_g_stream_manual_experiment.v2' and packet['arm']in('control','overlap')
    assert sha(packet['backend_packet'])==packet['backend_packet_sha256']
    for name,digest in packet['entry_files'].items():assert sha(Path(__file__).parent/name)==digest
    assert set(packet['entry_files'])=={'manual_entry.py','stream_execution.py'}
    from distillschedule.strategies import wan_worker as worker
    from distillschedule.runtime import wan13b_fa3_manual_experiment as fa3
    head=subprocess.check_output(['git','-C',str(worker.repository_root()),'rev-parse','HEAD'],text=True).strip()
    assert head==packet['source_head']
    assert str(packet['backend_packet'])==rest[rest.index('--packet')+1]
    assert packet['backend_packet_sha256']==rest[rest.index('--packet-sha256')+1]
    with explicit_worker(worker,enabled=packet['arm']=='overlap',entry_sha256=packet['entry_files']['stream_execution.py'],packet_sha256=a.stream_packet_sha256):
        fa3.main(rest)

if __name__=='__main__':main()
