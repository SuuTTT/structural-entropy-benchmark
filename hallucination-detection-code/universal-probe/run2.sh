cd /root/uprobe || exit 1
export HF_HOME=/root/.cache/huggingface HF_HUB_ENABLE_HF_TRANSFER=1 TOKENIZERS_PARALLELISM=false
for i in $(seq 1 180); do [ -f /root/uprobe/para_tqa.npz ] && break; sleep 10; done
echo "paraphrases ready $(date -u)" > /root/uprobe/logs2/run2.log
for M in qwen7b qwen7bi llama8b llama8bi; do for D in triviaqa nq squad sciq tqa; do
  [ -f /root/uprobe/raw2/${M}_${D}.npz ] && { echo "skip ${M}_${D}">>/root/uprobe/logs2/run2.log; continue; }
  echo "GEN2 ${M}_${D} $(date -u)">>/root/uprobe/logs2/run2.log
  python gen2.py $M $D 300 > /root/uprobe/logs2/gen2_${M}_${D}.log 2>&1
done; done
echo "JUDGE $(date -u)">>/root/uprobe/logs2/run2.log; python judge.py > /root/uprobe/logs2/judge.log 2>&1
echo "SE2 $(date -u)">>/root/uprobe/logs2/run2.log; python se2.py > /root/uprobe/logs2/se2.log 2>&1
touch /root/uprobe/V2_DONE; echo DONE >> /root/uprobe/logs2/run2.log
