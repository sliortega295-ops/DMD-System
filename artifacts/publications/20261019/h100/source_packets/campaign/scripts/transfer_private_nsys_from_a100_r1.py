"""Read-only validated A100 tool -> private H100 tool; no local payload."""
import importlib.util
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('existing_stream_transfer',HERE/'transfer_w6_steady_sources_r1.py')
existing=importlib.util.module_from_spec(spec);spec.loader.exec_module(existing)
ROOT='/home/jiangzhikun/yongyan_liu/Helios-H100'
destination=ROOT+'/tools/nsight_systems_2024.6.2_20261014_r1'
source='/share/project/shuchenweng/yongyan_liu/DistillSchedule_tools/nsight_systems'
receipt=existing.transfer('private_nsys_tool_2024_6_2_r1',source,['2024.6.2.225'],destination)
check=f'''set -eu
T={destination}/2024.6.2.225
test "$(sha256sum "$T/target-linux-x64/nsys" | cut -d' ' -f1)" = c7a1764113e90b450537ad10e051ca5e47849e50fee964d291212ca6c7277b0a
test "$(sha256sum "$T/host-linux-x64/QdstrmImporter" | cut -d' ' -f1)" = be5e3266faa2cc520fb226384c4b1597f0a477e7f6a936698766bb6a4532bb42
"$T/bin/nsys" --version
"$T/bin/nsys" profile --help > {ROOT}/research_artifacts/five_workload_h100_20260930/h100_resident_baselines_20261014_r1/logs/private_nsys_profile_help_r1.txt
'''
output=subprocess.check_output([*existing.SSH,existing.TARGET,check],text=True,stderr=subprocess.STDOUT,timeout=60)
assert '2024.6.2.225' in output
receipt.update(status='PASS_PRIVATE_TOOL_VERSION_AND_TWO_BINARY_IDENTITIES',
               version_output=output,advertised_source_bytes=975175627,
               installed_system_nsys_changed=False,training_environment_changed=False,
               GPU_capture='NOT_RUN_PENDING_BOUNDED_COMPATIBILITY_CANARY',
               nsys_sha256='c7a1764113e90b450537ad10e051ca5e47849e50fee964d291212ca6c7277b0a',
               importer_sha256='be5e3266faa2cc520fb226384c4b1597f0a477e7f6a936698766bb6a4532bb42')
p=HERE.parent/'manifests/private_nsys_tool_transfer_r1.json'
with p.open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
print(json.dumps(receipt))
