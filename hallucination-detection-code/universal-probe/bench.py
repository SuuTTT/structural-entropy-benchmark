import numpy as np,glob,os,math,sys
from sklearn.metrics import roc_auc_score as AUC, average_precision_score as AP
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
rng=np.random.RandomState(0)
UNC={'triviaqa','nq','squad','sciq'}; CW={'tqa'}
cells={}
for f in glob.glob('/root/uprobe/feats/*.npz'):
    name=f.split('/')[-1].replace('.npz',''); i=name.rfind('_'); m,ds=name[:i],name[i+1:]
    d=np.load(f); cells[(m,ds)]=dict(d=d,lab=d['labels'].astype(int),SE=d['SE'],AG=d['AG'],layers=[int(x) for x in d['layers']])
models=sorted(set(m for m,_ in cells))
def z(x): x=np.asarray(x,float); return (x-x.mean())/(x.std()+1e-9)
def split(n): te=np.arange(int(0.7*n),n); tr=np.arange(0,int(0.7*n)); return tr,te
def aurac(l,s):
    o=np.argsort(s); l=l[o]; return float(np.mean([l[i:].mean() for i in range(len(l))]))
def ci(l,s,B=1500):
    v=[];n=len(l)
    for _ in range(B):
        b=rng.randint(0,n,n)
        if len(set(l[b]))>1: v.append(AUC(l[b],s[b]))
    return (np.percentile(v,2.5),np.percentile(v,97.5)) if v else (0,0)
def build_probe(m,dslist,layers):
    best=None
    for L in layers:
        Xtr=[];ytr=[];Xva=[];yva=[]
        for ds in dslist:
            if (m,ds) not in cells: continue
            c=cells[(m,ds)]; tr,_=split(len(c['lab'])); rng.shuffle(tr); nv=max(15,int(0.2*len(tr)))
            Xva.append(c['d']['X_L%d'%L][tr[:nv]]); yva.append(c['lab'][tr[:nv]])
            Xtr.append(c['d']['X_L%d'%L][tr[nv:]]); ytr.append(c['lab'][tr[nv:]])
        if not Xtr: return None
        Xtr=np.vstack(Xtr);ytr=np.concatenate(ytr);Xva=np.vstack(Xva);yva=np.concatenate(yva)
        sc=StandardScaler().fit(Xtr)
        for C in [0.01,0.1,1.0]:
            clf=LogisticRegression(C=C,max_iter=2000,class_weight='balanced').fit(sc.transform(Xtr),ytr)
            try: a=AUC(yva,clf.predict_proba(sc.transform(Xva))[:,1])
            except: a=0.5
            if best is None or a>best[0]: best=(a,L,sc,clf)
    return best
out=open('/root/uprobe/BENCH_UNIV.txt','w')
def P(*a): print(*a); print(*a,file=out)
for m in models:
    dss=[ds for mm,ds in cells if mm==m]; layers=cells[(m,dss[0])]['layers']
    uncds=[d for d in dss if d in UNC]; cwds=[d for d in dss if d in CW]
    up=build_probe(m,uncds,layers); cp=build_probe(m,cwds,layers) if cwds else up
    if up is None: continue
    P('#### MODEL',m,'| unc-probe L%d valAUC%.3f'%(up[1],up[0]),'| cw-probe',('L%d valAUC%.3f'%(cp[1],cp[0]) if cwds else 'NA(use unc)'))
    rows={}; deltas=[]
    for ds in dss:
        c=cells[(m,ds)]; _,te=split(len(c['lab'])); lab=c['lab'][te]
        if len(set(lab))<2: continue
        upe=up[3].predict_proba(up[2].transform(c['d']['X_L%d'%up[1]][te]))[:,1]
        cpe=cp[3].predict_proba(cp[2].transform(c['d']['X_L%d'%cp[1]][te]))[:,1]
        se=c['SE'][te]; se=se if AUC(lab,se)>=.5 else -se
        ag=c['AG'][te]; agn=(ag-ag.min())/(ag.max()-ag.min()+1e-9)
        routed=agn*z(cpe)+(1-agn)*z(se)
        uvauc=up[0]; maxp=np.maximum(z(se),z(upe)); gated=maxp if uvauc>0.76 else z(se); max3=np.maximum(maxp,z(cpe)); m_={'SE':se,'unc_probe':upe,'cw_probe':cpe,'routed':routed,'se+unc_max':maxp,'gated_union':gated,'max3':max3}
        rows[ds]={k:AUC(lab,np.asarray(v,float)) for k,v in m_.items()}
        gg=np.asarray(gated,float); ss=np.asarray(se,float); d0=AUC(lab,gg)-AUC(lab,ss); vv=[]
        for _ in range(1500):
            bi=rng.randint(0,len(lab),len(lab))
            if len(set(lab[bi]))>1: vv.append(AUC(lab[bi],gg[bi])-AUC(lab[bi],ss[bi]))
        dlo,dhi=np.percentile(vv,[2.5,97.5]); deltas.append((ds,d0,dlo,dhi))
        lo,hi=ci(lab,np.asarray(routed,float))
        P('  %-9s reg=%s n=%d pos=%.2f | SE=%.3f unc=%.3f cw=%.3f ROUTED=%.3f[%.2f,%.2f] maxSEunc=%.3f'%(ds,'U' if ds in UNC else 'C',len(te),lab.mean(),rows[ds]['SE'],rows[ds]['unc_probe'],rows[ds]['cw_probe'],rows[ds]['routed'],lo,hi,rows[ds]['se+unc_max']))
    if rows:
        for k in ['SE','gated_union']:
            P('   AVG %-9s = %.4f'%(k,np.mean([rows[ds][k] for ds in rows])))
        P('   gated-SE per-cell delta: '+' '.join('%s=%+.3f[%+.2f,%+.2f]'%(d[0],d[1],d[2],d[3]) for d in deltas))
        sig=sum(1 for d in deltas if d[2]>0); P('   >> mean delta=%+.4f | cells gated>SE(CI>0)=%d/%d'%(np.mean([d[1] for d in deltas]),sig,len(deltas)))
out.close()
