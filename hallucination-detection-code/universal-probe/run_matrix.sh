cd /root/uprobe || exit 1
export HF_HOME=/root/.cache/huggingface HF_HUB_ENABLE_HF_TRANSFER=1 TOKENIZERS_PARALLELISM=false
for i in $(seq 1 60); do grep -q DEPS_DONE /root/uprobe/logs/pip.log 2>/dev/null && break; sleep 5; done
echo "deps ready $(date -u)" > /root/uprobe/logs/matrix.log
for M in qwen7b qwen7bi llama8b llama8bi; do
  for D in triviaqa nq squad sciq tqa; do
    if [ -f /root/uprobe/feats/${M}_${D}.npz ]; then echo "skip ${M}_${D}" >> /root/uprobe/logs/matrix.log; continue; fi
    echo "START ${M}_${D} $(date -u)" >> /root/uprobe/logs/matrix.log
    python prep.py $M $D 500 > /root/uprobe/logs/${M}_${D}.log 2>&1
    echo "END ${M}_${D} $(date -u) rc=$?" >> /root/uprobe/logs/matrix.log
  done
done
touch /root/uprobe/MATRIX_DONE
