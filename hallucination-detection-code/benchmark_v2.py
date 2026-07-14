import os,sys,math,numpy as np,torch
os.environ.setdefault('HF_HOME','/workspace/.hf_home')
from collections import Counter
from sklearn.metrics import roc_auc_score as AUC, average_precision_score as AP
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
m=sys.argv[1]; DS=sys.argv[2].split(','); rng=np.random.RandomState(0)
# ---- load feats + splits ----
cell={}
for ds in DS:
    d=np.load(f'/root/tsv/feats_{m}_{ds}.npz'); lab=d['labels'].astype(int); kept=d['kept']
    idx=np.load(f'/root/tsv/data_indices/data_index_{ds}.npy'); Nds=len(idx); wild=set(idx[:int(0.75*Nds)].tolist())
    te=np.array([k not in wild for k in kept]); cell[ds]=dict(d=d,lab=lab,kept=kept,te=te)
# ---- SE (cached) ----
def load_nli():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok=AutoTokenizer.from_pretrained('microsoft/deberta-v2-xlarge-mnli')
    nli=AutoModelForSequenceClassification.from_pretrained('microsoft/deberta-v2-xlarge-mnli',torch_dtype=torch.float16).cuda().eval()
    ENT=[k for k,v in nli.config.id2label.items() if 'entail' in v.lower()][0]
    return tok,nli,ENT
nli_state=[None]
def se_for(ds,kept):
    p=f'/root/tsv/se_{m}_{ds}.npz'
    if os.path.exists(p):
        z=np.load(p); return z['SE'],z['NC'],z['AG']
    if nli_state[0] is None: nli_state[0]=load_nli()
    tok,nli,ENT=nli_state[0]
    @torch.no_grad()
    def ent(a,b):
        x=tok(a,b,return_tensors='pt',truncation=True,max_length=256).to('cuda'); return nli(**x).logits.float().softmax(-1)[0][ENT].item()>0.5
    SE=np.zeros(len(kept));NC=np.ones(len(kept));AG=np.ones(len(kept))
    for j,qi in enumerate(kept):
        f=f'/root/tsv/save_for_eval/{ds}_hal_det/answers/batch_generations_hal_det_{m}_{ds}_answers_index_{int(qi)}.npy'
        if not os.path.exists(f): continue
        g=[str(x).strip() for x in np.load(f,allow_pickle=True).reshape(-1).tolist() if str(x).strip()]
        if not g: continue
        reps=[];asg=[]
        for x in g:
            pp=-1
            for ci,r in enumerate(reps):
                if ent(x,r) and ent(r,x): pp=ci;break
            if pp<0: reps.append(x);pp=len(reps)-1
            asg.append(pp)
        c=Counter(asg);n=len(asg); SE[j]=-sum((v/n)*math.log(v/n) for v in c.values()); NC[j]=len(reps); AG[j]=max(c.values())/n
    np.savez(p,SE=SE,NC=NC,AG=AG); return SE,NC,AG
for ds in DS:
    SE,NC,AG=se_for(ds,cell[ds]['kept']); cell[ds]['SE']=np.nan_to_num(SE); cell[ds]['NC']=NC; cell[ds]['AG']=AG
layers=cell[DS[0]]['d']['layers']
# ---- diversity (pooled) probe: select layer/C on pooled val ----
best=None
for L in layers:
    Xtr=[];ytr=[];Xva=[];yva=[]
    for ds in DS:
        c=cell[ds]; tp=np.where(~c['te'])[0]; rng.shuffle(tp); nv=max(15,int(0.2*len(tp)))
        Xva.append(c['d'][f'X_L{L}'][tp[:nv]]); yva.append(c['lab'][tp[:nv]])
        Xtr.append(c['d'][f'X_L{L}'][tp[nv:]]); ytr.append(c['lab'][tp[nv:]])
    Xtr=np.vstack(Xtr);ytr=np.concatenate(ytr);Xva=np.vstack(Xva);yva=np.concatenate(yva)
    scc=StandardScaler().fit(Xtr)
    for C in [0.01,0.1,1.0]:
        clf=LogisticRegression(C=C,max_iter=3000,class_weight='balanced').fit(scc.transform(Xtr),ytr)
        a=AUC(yva,clf.predict_proba(scc.transform(Xva))[:,1])
        if best is None or a>best[0]: best=(a,L,C)
_,L,C=best
# fit final pooled probe on ALL trainpool, and pooled-fusion LR
Xtr=[];ytr=[];ptr=[];setr=[];ncr=[];agr=[]
for ds in DS:
    c=cell[ds]; tp=np.where(~c['te'])[0]
    Xtr.append(c['d'][f'X_L{L}'][tp]); ytr.append(c['lab'][tp]); setr.append(c['SE'][tp]); ncr.append(c['NC'][tp]); agr.append(c['AG'][tp])
Xtr=np.vstack(Xtr);ytr=np.concatenate(ytr)
scc=StandardScaler().fit(Xtr); probe=LogisticRegression(C=C,max_iter=3000,class_weight='balanced').fit(scc.transform(Xtr),ytr)
ptr=probe.predict_proba(scc.transform(Xtr))[:,1]; setr=np.concatenate(setr); ncr=np.concatenate(ncr); agr=np.concatenate(agr)
def zt(x): return (x-x.mean())/(x.std()+1e-9)
Fn=np.column_stack([zt(ptr),zt(setr)]); Fg=np.column_stack([zt(ptr),zt(setr),zt(setr*0+setr),zt(ncr),zt(agr)])
fuseN=LogisticRegression(max_iter=3000,class_weight='balanced').fit(Fn,ytr)
fuseG=LogisticRegression(max_iter=3000,class_weight='balanced').fit(Fg,ytr)
def aurac(lab,sc):
    o=np.argsort(sc);lab=lab[o];return float(np.mean([lab[i:].mean() for i in range(len(lab))]))
def ci(lab,sc,B=2000):
    v=[];n=len(lab)
    for _ in range(B):
        bi=rng.randint(0,n,n)
        if len(set(lab[bi]))>1: v.append(AUC(lab[bi],sc[bi]))
    return np.percentile(v,2.5),np.percentile(v,97.5)
print(f'#### MODEL {m} | pooled probe L{L} C{C} valAUC{best[0]:.3f} | datasets {DS}')
for ds in DS:
    c=cell[ds]; te=np.where(c['te'])[0]; lab=c['lab'][te]
    Xte=scc.transform(c['d'][f'X_L{L}'][te]); pte=probe.predict_proba(Xte)[:,1]; sete=c['SE'][te]
    fn=fuseN.predict_proba(np.column_stack([zt(pte),zt(sete)]))[:,1]
    fg=fuseG.predict_proba(np.column_stack([zt(pte),zt(sete),zt(sete),zt(c['NC'][te]),zt(c['AG'][te])]))[:,1]
    se_o=sete if AUC(lab,sete)>=.5 else -sete
    agte=c["AG"][te]; agn=(agte-agte.min())/(agte.max()-agte.min()+1e-9)
    zp=(pte-pte.mean())/(pte.std()+1e-9); zs=(se_o-se_o.mean())/(se_o.std()+1e-9)
    seN=(sete-sete.min())/(sete.max()-sete.min()+1e-9)
    gate_agree=agn*zp+(1-agn)*zs
    gate_semag=seN*zs+(1-seN)*zp
    outs={'divprobe':pte,'SE':se_o,'naive_fuse':fn,'gate_agree':gate_agree,'gate_semag':gate_semag}
    print(f'-- {ds} test n={len(te)} pos={lab.mean():.2f} --')
    line=f'{m}/{ds}:'
    for k,s in outs.items():
        s=np.asarray(s,float); lo,hi=ci(lab,s); print(f'   {k:12s} AUROC={AUC(lab,s):.4f} [{lo:.3f},{hi:.3f}] PR={AP(lab,s):.3f} AURAC={aurac(lab,s):.3f}'); line+=f' {k}={AUC(lab,s):.4f}'
    open('/root/tsv/BENCH_V2.txt','a').write(line+'\n')
