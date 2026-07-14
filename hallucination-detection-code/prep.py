import os,sys,re,string,math,json,numpy as np,torch
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from datasets import load_dataset
m=sys.argv[1]; ds=sys.argv[2]; N=int(sys.argv[3]) if len(sys.argv)>3 else 500; K=10
HF={'qwen7b':'Qwen/Qwen2.5-7B','qwen7bi':'Qwen/Qwen2.5-7B-Instruct','llama8b':'NousResearch/Meta-Llama-3.1-8B','llama8bi':'NousResearch/Meta-Llama-3.1-8B-Instruct','mistral7b':'mistralai/Mistral-7B-v0.3','mistral7bi':'mistralai/Mistral-7B-Instruct-v0.3'}
def norm(s):
    s=s.lower().strip(); s=''.join(c for c in s if c not in string.punctuation)
    s=re.sub(r'\b(a|an|the)\b',' ',s); return re.sub(r'\s+',' ',s).strip()
def match(a,golds):
    a=norm(a)
    for g in golds:
        g=norm(str(g))
        if g and (g in a or a in g): return 1
    return 0
def load_ds(ds):
    if ds=='triviaqa': d=load_dataset('trivia_qa','rc.nocontext',split='validation'); return [(x['question'],list(x['answer']['aliases'])) for x in d.select(range(min(N,len(d))))]
    if ds=='nq': d=load_dataset('google-research-datasets/nq_open',split='validation'); return [(x['question'],list(x['answer'])) for x in d.select(range(min(N,len(d))))]
    if ds=='squad': d=load_dataset('rajpurkar/squad',split='validation'); return [(x['question'],list(x['answers']['text'])) for x in d.select(range(min(N,len(d))))]
    if ds=='sciq': d=load_dataset('allenai/sciq',split='validation'); return [(x['question'],[x['correct_answer']]) for x in d.select(range(min(N,len(d))))]
    if ds=='tqa': d=load_dataset('truthful_qa','generation')['validation']; return [(x['question'],[x['best_answer']]+list(x['correct_answers'])) for x in d.select(range(min(N,len(d))))]
    raise ValueError(ds)
data=load_ds(ds); path=HF[m]
tok=AutoTokenizer.from_pretrained(path); tok.pad_token=tok.pad_token or tok.eos_token
model=AutoModelForCausalLM.from_pretrained(path,torch_dtype=torch.float16,device_map='auto').eval()
nL=model.config.num_hidden_layers; layers=sorted(set(int(nL*f) for f in (0.4,0.55,0.7,0.85)))
print(m,ds,'nL',nL,'layers',layers,'N',len(data),flush=True)
feats={L:[] for L in layers}; labels=[]; samples=[]
for i,(q,golds) in enumerate(data):
    p=f'Answer the question concisely. Q: {q} A:'
    ids=tok(p,return_tensors='pt').input_ids.cuda()
    with torch.no_grad():
        g=model.generate(ids,max_new_tokens=48,do_sample=False,num_beams=1,pad_token_id=tok.eos_token_id)
        greedy=tok.decode(g[0,ids.shape[1]:],skip_special_tokens=True).strip()
        sq=model.generate(ids,max_new_tokens=48,do_sample=True,temperature=0.7,top_p=0.9,num_return_sequences=K,pad_token_id=tok.eos_token_id)
        samps=[tok.decode(sq[j,ids.shape[1]:],skip_special_tokens=True).strip() for j in range(K)]
        fp=tok(f'{p}{greedy}',return_tensors='pt').input_ids.cuda()
        out=model(fp,output_hidden_states=True)
        for L in layers: feats[L].append(out.hidden_states[L][0,-1,:].float().cpu().numpy())
    labels.append(match(greedy,golds)); samples.append(samps)
    if i%100==0: print(i,'lab',round(np.mean(labels),3),flush=True)
del model; torch.cuda.empty_cache()
# ---- SE via NLI ----
nt=AutoTokenizer.from_pretrained('microsoft/deberta-v2-xlarge-mnli'); nli=AutoModelForSequenceClassification.from_pretrained('microsoft/deberta-v2-xlarge-mnli',torch_dtype=torch.float16).cuda().eval()
ENT=[k for k,v in nli.config.id2label.items() if 'entail' in v.lower()][0]
@torch.no_grad()
def ent(a,b):
    x=nt(a,b,return_tensors='pt',truncation=True,max_length=256).to('cuda'); return nli(**x).logits.float().softmax(-1)[0][ENT].item()>0.5
SE=np.zeros(len(samples)); NC=np.ones(len(samples)); AG=np.ones(len(samples))
for j,gs in enumerate(samples):
    gs=[x for x in gs if x.strip()]
    if not gs: continue
    reps=[];asg=[]
    for x in gs:
        pp=-1
        for ci,r in enumerate(reps):
            if ent(x,r) and ent(r,x): pp=ci;break
        if pp<0: reps.append(x);pp=len(reps)-1
        asg.append(pp)
    c=Counter(asg);n=len(asg); SE[j]=-sum((v/n)*math.log(v/n) for v in c.values()); NC[j]=len(reps); AG[j]=max(c.values())/n
out={f'X_L{L}':np.stack(feats[L]) for L in layers}; out.update(labels=np.array(labels),SE=SE,NC=NC,AG=AG,layers=np.array(layers))
np.savez(f'/root/uprobe/feats/{m}_{ds}.npz',**out)
print('SAVED',m,ds,'n',len(labels),'pos',round(float(np.mean(labels)),3),'meanSE',round(float(SE.mean()),3),flush=True)
