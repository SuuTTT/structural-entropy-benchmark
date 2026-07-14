import numpy as np, os, math, torch
from collections import Counter
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os as _o; _o.environ['HF_HOME']='/workspace/.hf_home'
tok=AutoTokenizer.from_pretrained('microsoft/deberta-v2-xlarge-mnli')
nli=AutoModelForSequenceClassification.from_pretrained('microsoft/deberta-v2-xlarge-mnli',torch_dtype=torch.float16).cuda().eval()
id2l={int(k):v.lower() for k,v in nli.config.id2label.items()}
ENT=[k for k,v in id2l.items() if 'entail' in v][0]
print('id2label',id2l,'ENT',ENT)
@torch.no_grad()
def ent(a,b):
    x=tok(a,b,return_tensors='pt',truncation=True,max_length=256).to('cuda')
    return nli(**x).logits.float().softmax(-1)[0][ENT].item()>0.5
def sem_ent(gens):
    gens=[g.strip() for g in gens if g and g.strip()]
    if not gens: return np.nan
    reps=[]; asg=[]
    for g in gens:
        p=-1
        for ci,r in enumerate(reps):
            if ent(g,r) and ent(r,g): p=ci; break
        if p<0: reps.append(g); p=len(reps)-1
        asg.append(p)
    c=Counter(asg); n=len(asg)
    return -sum((v/n)*math.log(v/n) for v in c.values())
tsv=np.load('/root/tsv/test_scores.npy').astype(float); lab=np.load('/root/tsv/test_labels.npy').astype(int)
idx=np.load('/root/tsv/data_indices/data_index_nq_open.npy'); wild=set(idx[:300].tolist())
test_idx=[i for i in range(400) if i not in wild]
print('len(test_idx)',len(test_idx),'len(tsv)',len(tsv),'len(lab)',len(lab))
ans='/root/tsv/save_for_eval/nq_open_hal_det/answers'; se=[]
for qi in test_idx:
    f=f'{ans}/batch_generations_hal_det_qwen2.5-7B_nq_open_answers_index_{qi}.npy'
    if not os.path.exists(f): se.append(np.nan); continue
    g=np.load(f,allow_pickle=True).tolist(); se.append(sem_ent([str(x) for x in g]))
se=np.array(se); m=~np.isnan(se); tsv,se2,lab2=tsv[m],se[m],lab[m]
if roc_auc_score(lab2,tsv)<0.5: tsv=-tsv
if roc_auc_score(lab2,se2)<0.5: se2=-se2
def z(x): return (x-x.mean())/(x.std()+1e-9)
def cv(X):
    skf=StratifiedKFold(5,shuffle=True,random_state=0); oof=np.zeros(len(lab2))
    for tr,te in skf.split(X,lab2): oof[te]=LogisticRegression(max_iter=500).fit(X[tr],lab2[tr]).predict_proba(X[te])[:,1]
    return roc_auc_score(lab2,oof)
a_t=roc_auc_score(lab2,tsv); a_s=roc_auc_score(lab2,se2); co=np.corrcoef(tsv,se2)[0,1]
cvt=cv(z(tsv).reshape(-1,1)); cvf=cv(np.column_stack([z(tsv),z(se2)]))
out=f'N={m.sum()}  TSV={a_t:.4f}  NLI-SE={a_s:.4f}  corr={co:.3f}  |  logregCV TSV={cvt:.4f}  TSV+NLI-SE={cvf:.4f}  delta={cvf-cvt:+.4f}'
print(out); open('/root/tsv/SE_FUSION_NQ_RESULT.txt','a').write('\nnq_open NLI-SE: '+out+'\n')
