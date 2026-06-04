#!/usr/bin/env bash
# Dispatch the SE benchmark to a rented box. Run from the AWS control box.
#   bash dispatch.sh <ssh_host> <ssh_port> <stage>
# stages: setup | community | dese | all
# SSH key + repos are assumed at the defaults below.
set -uo pipefail
HOST=${1:?ssh host}; PORT=${2:?ssh port}; STAGE=${3:-all}
KEY=${GPU_FLEET_KEY:-~/.ssh/vastai_id_ed25519}
REPOS=~/se-bench-repos
SUITE=/home/ubuntu/structural-entropy-survey-clean/benchmark/suite

# pick the working port (vast sometimes reports proxy port; real sshd at +1)
ssh_ok() { timeout 10 ssh -i $KEY -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=8 -p "$1" root@$HOST "true" 2>/dev/null; }
if ssh_ok "$PORT"; then P=$PORT; elif ssh_ok $((PORT+1)); then P=$((PORT+1)); else echo "SSH not reachable on $PORT/$((PORT+1))"; exit 1; fi
SSH="ssh -i $KEY -o StrictHostKeyChecking=no -p $P root@$HOST"
RSYNC="rsync -az -e \"ssh -i $KEY -o StrictHostKeyChecking=no -p $P\""
echo "== using $HOST:$P =="

sync_code() {
  echo "== rsync harness + repos =="
  $SSH "mkdir -p /root/se-bench/harness /root/se-bench/results/community_detection"
  eval $RSYNC --exclude='.git' "$SUITE/harness/" "root@$HOST:/root/se-bench/harness/"
  eval $RSYNC --exclude='.git' "$REPOS/" "root@$HOST:/root/se-bench-repos/"
}

case "$STAGE" in
  setup|all)
    sync_code
    echo "== install system + harness deps =="
    $SSH "apt-get update -q && DEBIAN_FRONTEND=noninteractive apt-get install -y default-jre-headless unrar >/dev/null 2>&1; \
          pip install -q -r /root/se-bench/harness/requirements.txt; \
          cd /root/se-bench-repos/deDoc && unrar x -o+ deDoc.rar >/dev/null 2>&1; ls -1 *.jar; \
          nvidia-smi --query-gpu=name --format=csv,noheader"
    ;;&
  community|all)
    echo "== launch baselines + deDoc (CPU) detached =="
    $SSH "mkdir -p /root/se-bench/results/community_detection; \
      (cd /root/se-bench/harness && nohup python3 -u run_community_baselines.py --out /root/se-bench/results/community_detection > /root/se-bench/baselines.log 2>&1 </dev/null &); \
      (cd /root/se-bench/harness && nohup python3 -u run_dedoc.py --jar /root/se-bench-repos/deDoc/deDoc.jar --out /root/se-bench/results/community_detection > /root/se-bench/dedoc.log 2>&1 </dev/null &); \
      echo launched"
    ;;&
  dese)
    echo "== DeSE env + run (GPU) — install cu118 stack first =="
    $SSH "pip install -q munkres matplotlib scipy 2>&1 | tail -1; \
      pip install -q torch_geometric==2.5.3 2>&1 | tail -1; \
      pip install -q torch_scatter -f https://data.pyg.org/whl/torch-2.3.1+cu118.html 2>&1 | tail -1; \
      pip install -q dgl -f https://data.dgl.ai/wheels/torch-2.3/cu118/repo.html 2>&1 | tail -1; \
      python3 -c 'import torch_scatter,dgl,torch_geometric,munkres;print(\"deps ok\")' || echo DEP_FAIL; \
      cd /root/se-bench/harness && GPU_NAME=\$(nvidia-smi --query-gpu=name --format=csv,noheader) \
      nohup python3 -u run_dese.py --repo /root/se-bench-repos/DeSE --gpu 0 --out /root/se-bench/results/community_detection > /root/se-bench/dese.log 2>&1 </dev/null & echo launched"
    ;;
esac
echo "== done stage=$STAGE =="
