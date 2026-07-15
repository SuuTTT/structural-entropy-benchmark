import os,sys,glob,math,numpy as np,torch
os.environ.setdefault('HF_HOME','/root/.cache/huggingface')
from collections import Counter
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tok=AutoTokenizer.from_pretrained('microsoft/deberta-v2-xlarge-mnli'); nli=AutoModelForSequenceClassification.from_pretrained('microsoft/deberta-v2-xlarge-mnli',torch_dtype=torch.float16).cuda().eval()
ENT=[k for k,v in nli.config.id2label.items() if 'entail' in v.lower()][0]
@torch.no_grad()
def ent(a,b):
    x=tok(a,b,return_tensors='pt',truncation=True,max_length=256).to('cuda'); return nli(**x).logits.float().softmax(-1)[0][ENT].item()>0.5
def se(gens):
    gens=[str(g).strip() for g in gens if str(g).strip()]
    if not gens: return 0.0,1,1.0
    reps=[];asg=[]
    for g in gens:
        p=-1
        for ci,r in enumerate(reps):
            if ent(g,r) and ent(r,g): p=ci;break
        if p<0: reps.append(g);p=len(reps)-1
        asg.append(p)
    c=Counter(asg);n=len(asg); return -sum((v/n)*math.log(v/n) for v in c.values()),len(reps),max(c.values())/n
for f in sorted(glob.glob('/root/uprobe/raw2/*.npz')):
    nm=f.split('/')[-1].replace('.npz',''); out=f'/root/uprobe/feats2/se_{nm}.npz'
    if os.path.exists(out): continue
    d=np.load(f,allow_pickle=True); S=d['samples'];PA=d['para_ans'];GR=d['greedy']
    n=len(GR); ST=np.zeros(n);SP=np.zeros(n);AT=np.ones(n);AP=np.ones(n)
    for i in range(n):
        ST[i],_,AT[i]=se(list(S[i]))
        SP[i],_,AP[i]=se([str(GR[i])]+list(PA[i]))
    np.savez(out,SE_temp=ST,SE_para=SP,AG_temp=AT,AG_para=AP); print('SE2',nm,'meanST',round(float(ST.mean()),2),'meanSP',round(float(SP.mean()),2),flush=True)
