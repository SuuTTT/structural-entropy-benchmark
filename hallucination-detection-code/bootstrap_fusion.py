import numpy as np, os, math, torch
from collections import Counter
from sklearn.metrics import roc_auc_score as AUC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer, AutoModelForSequenceClassification
os.environ['HF_HOME']='/workspace/.hf_home'
tok=AutoTokenizer.from_pretrained('microsoft/deberta-v2-xlarge-mnli')
nli=AutoModelForSequenceClassification.from_pretrained('microsoft/deberta-v2-xlarge-mnli',torch_dtype=torch.float16).cuda().eval()
id2l={int(k):v.lower() for k,v in nli.config.id2label.items()}
ENT=[k for k,v in id2l.items() if 'entail' in v][0]
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
    c=Counter(asg); n=len(asg); return -sum((v/n)*math.log(v/n) for v in c.values())
def z(x): return (x-x.mean())/(x.std()+1e-9)
def se_for(ds):
    idx=np.load(f'/root/tsv/data_indices/data_index_{ds}.npy'); wild=set(idx[:300].tolist())
    test_idx=[i for i in range(400) if i not in wild]
    ans=f'/root/tsv/save_for_eval/{ds}_hal_det/answers'; out=[]
    for qi in test_idx:
        f=f'{ans}/batch_generations_hal_det_qwen2.5-7B_{ds}_answers_index_{qi}.npy'
        if not os.path.exists(f): out.append(np.nan); continue
        g=np.load(f,allow_pickle=True).tolist(); out.append(sem_ent([str(x) for x in g]))
    return np.array(out)
for ds in ['triviaqa','nq_open']:
    tsv=np.load(f'/root/tsv/test_scores_{ds}.npy').astype(float); lab=np.load(f'/root/tsv/test_labels_{ds}.npy').astype(int)
    se=se_for(ds); m=~np.isnan(se); tsv,se,lab=tsv[m],se[m],lab[m]
    if AUC(lab,tsv)<0.5: tsv=-tsv
    if AUC(lab,se)<0.5: se=-se
    X=np.column_stack([z(tsv),z(se)]); skf=StratifiedKFold(5,shuffle=True,random_state=0); oof=np.zeros(len(lab))
    for tr,te in skf.split(X,lab): oof[te]=LogisticRegression(max_iter=500).fit(X[tr],lab[tr]).predict_proba(X[te])[:,1]
    d_obs=AUC(lab,oof)-AUC(lab,tsv); rng=np.random.RandomState(0); ds_boot=[]; n=len(lab)
    for b in range(2000):
        bi=rng.randint(0,n,n)
        if len(set(lab[bi].tolist()))<2: continue
        ds_boot.append(AUC(lab[bi],oof[bi])-AUC(lab[bi],tsv[bi]))
    ds_boot=np.array(ds_boot); lo,hi=np.percentile(ds_boot,[2.5,97.5])
    line=f'{ds}: N={n} TSV={AUC(lab,tsv):.4f} SE={AUC(lab,se):.4f} fused={AUC(lab,oof):.4f} delta={d_obs:+.4f} 95%CI=[{lo:+.4f},{hi:+.4f}] frac>0={(ds_boot>0).mean():.3f}'
    print(line); open('/root/tsv/BOOTSTRAP_FUSION_RESULT.txt','a').write(line+chr(10))
