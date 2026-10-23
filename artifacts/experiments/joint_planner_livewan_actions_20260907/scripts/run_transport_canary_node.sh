#!/usr/bin/env bash
set -euo pipefail
node="$1"
root=/share/project/shuchenweng/yongyan_liu/DistillSchedule_runs/joint_planner_20260907/livewan_components_r1
repo=/share/project/shuchenweng/yongyan_liu/DMD-System_deploy/joint-livewan-actions-0b1b62d88254-20260907
out="$root/transport-canary-r1/node${node}"
mkdir "$out"
exec > >(tee "$out/node.log") 2>&1
exec 9>/tmp/distillschedule_gpu_node.lock
flock -n 9 || exit 73
date -u +%FT%TZ
hostname
test "$(git -C "$repo" rev-parse HEAD)" = 0b1b62d8825484d3153ce758bf33d6126856fee8
test -z "$(git -C "$repo" status --short)"
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,driver_version --format=csv,noheader
apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader)
printf 'COMPUTE_APPS_BEFORE=%s\n' "$apps"
test -z "$apps"
free -b
cat /sys/fs/cgroup/memory.events | tee "$out/memory.events.before"
cat /sys/fs/cgroup/memory.current
cat /sys/fs/cgroup/memory.max
findmnt -T "$root" -o TARGET,SOURCE,FSTYPE
if [ "$node" = 0 ]; then
  /tmp/distillschedule-helios-official-torch210/bin/python -c 'import socket;s=socket.socket();s.bind(("172.24.211.163",29937));s.close()'
fi
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$repo/src"
export NCCL_SOCKET_IFNAME=eth0 GLOO_SOCKET_IFNAME=eth0
cd "$repo"
set +e
timeout --kill-after=20s 150s /tmp/distillschedule-helios-official-torch210/bin/python \
 scripts/distillschedule/canary_livewan_action_observation.py \
 --node-rank "$node" --master-addr 172.24.211.163 --port 29937 \
 --expected-head 0b1b62d8825484d3153ce758bf33d6126856fee8 --output "$out/receipt.json"
code=$?
set -e
printf '%s\n' "$code" > "$out/exit-code.txt"
cat /sys/fs/cgroup/memory.events | tee "$out/memory.events.after"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | tee "$out/gpu-processes-after.txt"
test ! -s "$out/gpu-processes-after.txt" || exit 74
date -u +%FT%TZ
exit "$code"
