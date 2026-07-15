cd /root/uprobe || exit 1
export HF_HOME=/root/.cache/huggingface HF_HUB_ENABLE_HF_TRANSFER=1 TOKENIZERS_PARALLELISM=false
for i in $(seq 1 400); do [ -f /root/uprobe/V2_DONE ] && break; sleep 30; done
echo "run3 start $(date -u)" > /root/uprobe/logs2/run3.log
for M in qwen14b qwen14bi mistral7bi mistral7b yi9b; do for D in triviaqa nq squad sciq tqa; do
  [ -f /root/uprobe/raw2/${M}_${D}.npz ] && continue
  echo "GEN3 ${M}_${D} $(date -u)">>/root/uprobe/logs2/run3.log
  python gen2.py $M $D 300 > /root/uprobe/logs2/gen2_${M}_${D}.log 2>&1
done; done
echo "JUDGE3 $(date -u)">>/root/uprobe/logs2/run3.log; python judge.py > /root/uprobe/logs2/judge3.log 2>&1
echo "SE3 $(date -u)">>/root/uprobe/logs2/run3.log; python se2.py > /root/uprobe/logs2/se23f.log 2>&1
touch /root/uprobe/V3_DONE; echo DONE>>/root/uprobe/logs2/run3.log
