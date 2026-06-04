#!/usr/bin/env bash
# One-time setup on a freshly rented vast.ai box for the SE benchmark.
# Usage (on the box): bash setup_box.sh
set -e
echo "== system =="; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
java -version 2>&1 | head -1 || (apt-get update -q && apt-get install -y default-jre-headless)
echo "== harness deps =="
pip install -q -r /root/se-bench/harness/requirements.txt
echo "== sanity: import harness + objectives =="
cd /root/se-bench/harness && python3 - <<'PY'
import datasets, metrics
G, lab = datasets.sbm(150, 3, 0.30, 0.05)
import networkx as nx
print("SBM nodes/edges:", G.number_of_nodes(), G.number_of_edges())
print("cross_objective on ground truth:", metrics.cross_objective(G, lab))
PY
echo "== done =="
