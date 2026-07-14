import numpy as np, os, math
from collections import Counter
from sklearn.metrics import roc_auc_score

tsv = np.load('/root/tsv/test_scores.npy').astype(float)
lab = np.load('/root/tsv/test_labels.npy').astype(int)
index = np.load('/root/tsv/data_indices/data_index_tqa.npy')
L = 817
wild = set(index[:int(0.75*L)].tolist())
test_idx = [i for i in range(L) if i not in wild]
print('test_idx', len(test_idx), 'tsv', tsv.shape, 'lab', lab.shape, 'pos', int(lab.sum()))
ans='/root/tsv/save_for_eval/tqa_hal_det/answers'
def norm(s): return ' '.join(str(s).lower().strip().split())
se=[]
for qi in test_idx:
    f=f'{ans}/batch_generations_hal_det_qwen2.5-7B_tqa_answers_index_{qi}.npy'
    if not os.path.exists(f): se.append(np.nan); continue
    g=np.load(f, allow_pickle=True)
    g=[norm(x) for x in (g.tolist() if hasattr(g,'tolist') else list(g))]
    g=[x for x in g if x]
    if not g: se.append(np.nan); continue
    c=Counter(g); n=sum(c.values())
    se.append(-sum((v/n)*math.log(v/n) for v in c.values()))
se=np.array(se)
m=~np.isnan(se)
tsv,se2,lab2=tsv[m],se[m],lab[m]
def au(sc,l):
    a=roc_auc_score(l,sc); return (a,1) if a>=0.5 else (1-a,-1)
a_tsv,st=au(tsv,lab2); a_se,ss=au(se2,lab2)
def z(x): return (x-x.mean())/(x.std()+1e-9)
fused=z(st*tsv)+z(ss*se2)
a_fus=roc_auc_score(lab2,fused); a_fus=max(a_fus,1-a_fus)
corr=np.corrcoef(tsv,se2)[0,1]
print(f'N={m.sum()}  TSV-alone={a_tsv:.4f}  answerSE-alone={a_se:.4f}  FUSED={a_fus:.4f}  corr(TSV,SE)={corr:.3f}')
open('/root/tsv/SE_FUSION_RESULT.txt','w').write(f'TruthfulQA Qwen2.5-7B  N={m.sum()}\nTSV-alone AUROC={a_tsv:.4f}\nanswerSE-alone AUROC={a_se:.4f}\nFUSED(z-sum) AUROC={a_fus:.4f}\ncorr(TSV,SE)={corr:.3f}\nverdict={"FUSION BEATS TSV" if a_fus>a_tsv+0.005 else "no gain (TSV latent already captures it)"}\n')
