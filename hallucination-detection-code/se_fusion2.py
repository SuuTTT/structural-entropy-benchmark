import numpy as np, os, math
from collections import Counter
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
tsv=np.load('/root/tsv/test_scores.npy').astype(float); lab=np.load('/root/tsv/test_labels.npy').astype(int)
index=np.load('/root/tsv/data_indices/data_index_tqa.npy'); L=817
wild=set(index[:int(0.75*L)].tolist()); test_idx=[i for i in range(L) if i not in wild]
ans='/root/tsv/save_for_eval/tqa_hal_det/answers'
def norm(s): return ' '.join(str(s).lower().strip().split())
se=[]
for qi in test_idx:
    f=f'{ans}/batch_generations_hal_det_qwen2.5-7B_tqa_answers_index_{qi}.npy'
    if not os.path.exists(f): se.append(np.nan); continue
    g=np.load(f,allow_pickle=True); g=[norm(x) for x in (g.tolist() if hasattr(g,'tolist') else list(g))]; g=[x for x in g if x]
    if not g: se.append(np.nan); continue
    c=Counter(g); n=sum(c.values()); se.append(-sum((v/n)*math.log(v/n) for v in c.values()))
se=np.array(se); m=~np.isnan(se); tsv,se2,lab2=tsv[m],se[m],lab[m]
# orient
if roc_auc_score(lab2,tsv)<0.5: tsv=-tsv
if roc_auc_score(lab2,se2)<0.5: se2=-se2
def z(x): return (x-x.mean())/(x.std()+1e-9)
X=np.column_stack([z(tsv),z(se2)]); Xtsv=z(tsv).reshape(-1,1)
def cv_auroc(X):
    skf=StratifiedKFold(5,shuffle=True,random_state=0); oof=np.zeros(len(lab2))
    for tr,te in skf.split(X,lab2):
        lr=LogisticRegression().fit(X[tr],lab2[tr]); oof[te]=lr.predict_proba(X[te])[:,1]
    return roc_auc_score(lab2,oof)
a_tsv=roc_auc_score(lab2,tsv); a_se=roc_auc_score(lab2,se2)
cv_tsv=cv_auroc(Xtsv); cv_fus=cv_auroc(X)
print(f'N={m.sum()}  TSV-alone={a_tsv:.4f}  SE-alone={a_se:.4f}  |  logregCV TSV-only={cv_tsv:.4f}  logregCV TSV+SE={cv_fus:.4f}  delta={cv_fus-cv_tsv:+.4f}')
open('/root/tsv/SE_FUSION_RESULT.txt','a').write(f'\nLEARNED fusion (logreg 5fold CV): TSV-only={cv_tsv:.4f} TSV+SE={cv_fus:.4f} delta={cv_fus-cv_tsv:+.4f}\nSE-alone(string-cluster)={a_se:.4f} (weak; NLI-cluster would be fairer)\n')
