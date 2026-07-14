import os,sys,math,numpy as np,torch
os.environ.setdefault('HF_HOME','/workspace/.hf_home')
from collections import Counter
from sklearn.metrics import roc_auc_score as AUC, average_precision_score as AP
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer, AutoModelForSequenceClassification
model_name=sys.argv[1]; ds=sys.argv[2]; rng=np.random.RandomState(0)
d=np.load(f'/root/tsv/feats_{model_name}_{ds}.npz'); labels=d['labels'].astype(int); kept=d['kept']; layers=d['layers']
idx=np.load(f'/root/tsv/data_indices/data_index_{ds}.npy'); wild=set(idx[:300].tolist())
is_test=np.array([kept[j] not in wild for j in range(len(kept))])
tr_all=np.where(~is_test)[0]; te=np.where(is_test)[0]
rng.shuffle(tr_all); nval=max(20,int(0.2*len(tr_all))); va=tr_all[:nval]; tr=tr_all[nval:]
def aurac(lab,sc):
    o=np.argsort(sc); lab=lab[o]; n=len(lab); accs=[lab[i:].mean() for i in range(n)]; return float(np.mean(accs))
def boot(lab,sc,f=AUC,B=2000):
    v=[]; n=len(lab)
    for _ in range(B):
        bi=rng.randint(0,n,n)
        if len(set(lab[bi]))<2: continue
        v.append(f(lab[bi],sc[bi]))
    return float(np.percentile(v,2.5)),float(np.percentile(v,97.5))
# ---- SE via NLI over cached batch_generations ----
tok=AutoTokenizer.from_pretrained('microsoft/deberta-v2-xlarge-mnli')
nli=AutoModelForSequenceClassification.from_pretrained('microsoft/deberta-v2-xlarge-mnli',torch_dtype=torch.float16).cuda().eval()
ENT=[k for k,v in nli.config.id2label.items() if 'entail' in v.lower()][0]
@torch.no_grad()
def ent(a,b):
    x=tok(a,b,return_tensors='pt',truncation=True,max_length=256).to('cuda'); return nli(**x).logits.float().softmax(-1)[0][ENT].item()>0.5
def se_feats(qi):
    f=f'/root/tsv/save_for_eval/{ds}_hal_det/answers/batch_generations_hal_det_{model_name}_{ds}_answers_index_{qi}.npy'
    if not os.path.exists(f): return np.nan,np.nan,np.nan
    g=[str(x).strip() for x in np.load(f,allow_pickle=True).reshape(-1).tolist() if str(x).strip()]
    if not g: return np.nan,np.nan,np.nan
    reps=[];asg=[]
    for x in g:
        p=-1
        for ci,r in enumerate(reps):
            if ent(x,r) and ent(r,x): p=ci;break
        if p<0: reps.append(x);p=len(reps)-1
        asg.append(p)
    c=Counter(asg);n=len(asg); H=-sum((v/n)*math.log(v/n) for v in c.values()); return H,len(reps),max(c.values())/n
SE=np.zeros(len(kept)); NC=np.zeros(len(kept)); AG=np.zeros(len(kept))
for j in range(len(kept)):
    H,nc,ag=se_feats(int(kept[j])); SE[j]=H; NC[j]=nc if not np.isnan(nc) else 1; AG[j]=ag if not np.isnan(ag) else 1
SE=np.nan_to_num(SE)
# ---- probe: pick best layer/C on VAL ----
best=None
for L in layers:
    X=d[f'X_L{L}']; sc=StandardScaler().fit(X[tr])
    for C in [0.01,0.1,1.0]:
        clf=LogisticRegression(C=C,max_iter=2000,class_weight='balanced').fit(sc.transform(X[tr]),labels[tr])
        va_auc=AUC(labels[va],clf.predict_proba(sc.transform(X[va]))[:,1])
        if best is None or va_auc>best[0]: best=(va_auc,L,C,sc,clf)
_,L,C,sc,clf=best; X=d[f'X_L{L}']
probe_all=clf.predict_proba(sc.transform(X))[:,1]
# fusion features on train/test
def mat(cols,ref_idx): return np.column_stack([ (c-c[ref_idx].mean())/(c[ref_idx].std()+1e-9) for c in cols])
probe_score=probe_all
naive=mat([probe_score,SE],tr); adapt=mat([probe_score,SE,SE,NC,AG],tr)
def fuse(M):
    clf=LogisticRegression(max_iter=2000,class_weight='balanced').fit(M[tr],labels[tr]); return clf.predict_proba(M[te])[:,1]
res={}
res['probe(L%d)'%L]=probe_score[te]
res['SE']=SE[te]
res['naive_fuse']=fuse(naive)
res['adapt_fuse']=fuse(adapt)
# TSV baseline if aligned scores exist
import glob
def report():
    print(f'=== {model_name} / {ds} | test n={len(te)} pos={labels[te].mean():.2f} | probe layer L{L} C{C} valAUC{best[0]:.3f} ===')
    for k,s in res.items():
        s=np.asarray(s,float)
        if AUC(labels[te],s)<0.5 and k in ('SE',): s=-s
        lo,hi=boot(labels[te],s); 
        print(f'{k:14s} AUROC={AUC(labels[te],s):.4f} [{lo:.3f},{hi:.3f}]  PR-AUC={AP(labels[te],s):.4f}  AURAC={aurac(labels[te],s):.3f}')
    open('/root/tsv/BENCHMARK_RESULTS.txt','a').write(f'{model_name}/{ds} L{L} C{C}: '+' | '.join(f'{k}={AUC(labels[te],np.asarray(s,float)):.4f}' for k,s in res.items())+'\n')
report()
