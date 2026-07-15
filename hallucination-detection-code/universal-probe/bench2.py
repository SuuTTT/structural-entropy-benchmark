import numpy as np,glob,os
from sklearn.metrics import roc_auc_score as AUC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
rng=np.random.RandomState(0)
def zc(x):x=np.asarray(x,float);return (x-x.mean())/(x.std()+1e-9)
def complete(m):
    dss=[]
    for ds in ['triviaqa','nq','squad','sciq','tqa']:
        if os.path.exists(f'/root/uprobe/raw2/{m}_{ds}.npz') and os.path.exists(f'/root/uprobe/feats2/judge_{m}_{ds}.npy') and os.path.exists(f'/root/uprobe/feats2/se_{m}_{ds}.npz'): dss.append(ds)
    return dss
def loadm(m,dss):
    Ls=[int(x) for x in np.load(f'/root/uprobe/raw2/{m}_{dss[0]}.npz',allow_pickle=True)['layers']]
    X={L:[] for L in Ls};LAB=[];ST=[];SP=[];AGp=[]
    for ds in dss:
        r=np.load(f'/root/uprobe/raw2/{m}_{ds}.npz',allow_pickle=True); s=np.load(f'/root/uprobe/feats2/se_{m}_{ds}.npz'); j=np.load(f'/root/uprobe/feats2/judge_{m}_{ds}.npy')
        for L in Ls: X[L].append(r['X_L%d'%L])
        LAB.append(j.astype(int)); ST.append(s['SE_temp']); SP.append(s['SE_para']); AGp.append(s['AG_para'])
    for L in Ls: X[L]=np.vstack(X[L])
    return Ls,X,np.concatenate(LAB),np.concatenate(ST),np.concatenate(SP),np.concatenate(AGp)
def ci(lab,a,b,B=2000):
    d0=AUC(lab,a)-AUC(lab,b);v=[];n=len(lab)
    for _ in range(B):
        i=rng.randint(0,n,n)
        if len(set(lab[i]))>1: v.append(AUC(lab[i],a[i])-AUC(lab[i],b[i]))
    return d0,np.percentile(v,2.5),np.percentile(v,97.5)
out=open('/root/uprobe/BENCH2.txt','a')
def P(*a):print(*a);print(*a,file=out,flush=True)
for m in ['qwen7b','qwen7bi','llama8b','llama8bi']:
    dss=complete(m)
    if len(dss)<4: continue
    Ls,X,LAB,ST,SP,AGp=loadm(m,dss); n=len(LAB)
    idx=np.arange(n);rng.shuffle(idx);tr,te=idx[:int(.7*n)],idx[int(.7*n):]
    thr=np.median(ST[tr])
    set_c=lambda s: s if AUC(LAB[tr],s[tr])>=.5 else -s
    se_t=set_c(ST); se_p=set_c(SP)   # correctness-oriented (high=correct)
    # residual probe on confident (low SE_temp) trainpool
    ctr=tr[ST[tr]<thr]; rng.shuffle(ctr); nv=int(.2*len(ctr));cva,cft=ctr[:nv],ctr[nv:]
    best=None
    for L in Ls:
        sc=StandardScaler().fit(X[L][cft]); clf=LogisticRegression(C=0.1,max_iter=2000,class_weight='balanced').fit(sc.transform(X[L][cft]),LAB[cft])
        try:a=AUC(LAB[cva],clf.predict_proba(sc.transform(X[L][cva]))[:,1])
        except:a=.5
        if best is None or a>best[0]:best=(a,L,sc,clf)
    _,L,sc,clf=best; pr=clf.predict_proba(sc.transform(X[L]))[:,1]  # high=correct
    umax=np.maximum(zc(se_t),zc(se_p))
    conf=(ST<thr).astype(float)
    F=np.column_stack([zc(se_t),zc(se_p),zc(pr),conf,zc(se_t)*conf])
    _vsz=int(0.25*len(tr)); _r=tr.copy(); rng.shuffle(_r); _vs,_st=_r[:_vsz],_r[_vsz:]
    meta=LogisticRegression(max_iter=2000,class_weight="balanced").fit(F[_st],LAB[_st]); mp=meta.predict_proba(F)[:,1]
    gm = mp if AUC(LAB[_vs],mp[_vs])>AUC(LAB[_vs],se_t[_vs]) else se_t
    P('#### %s n_te=%d pos=%.2f | baseline SE_temp=%.4f'%(m,len(te),LAB[te].mean(),AUC(LAB[te],se_t[te])))
    for nm,sig in [('META',mp),('GATED_META',gm)]:
        d,lo,hi=ci(LAB[te],sig[te],se_t[te]); P('   %-13s AUROC=%.4f  d_vs_SEtemp=%+.4f[%+.3f,%+.3f]'%(nm,AUC(LAB[te],sig[te]),d,lo,hi))
out.close()
