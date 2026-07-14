import os, sys, re, string, numpy as np, torch
os.environ.setdefault('HF_HOME','/workspace/.hf_home'); os.environ.setdefault('HF_HUB_OFFLINE','1'); os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
HF={'qwen2.5-7B':'Qwen/Qwen2.5-7B','qwen2.5-7B-instruct':'Qwen/Qwen2.5-7B-Instruct','llama3.1-8B':'NousResearch/Meta-Llama-3.1-8B'}
model_name=sys.argv[1]; ds=sys.argv[2]
path=HF[model_name]
def norm(s):
    s=s.lower().strip(); s=''.join(c for c in s if c not in string.punctuation)
    s=re.sub(r'\b(a|an|the)\b',' ',s); return re.sub(r'\s+',' ',s).strip()
def match(ans, golds):
    a=norm(ans)
    for g in golds:
        g=norm(g)
        if g and (g in a or a in g): return 1
    return 0
if ds=='triviaqa': data=load_dataset('trivia_qa','rc.nocontext',split='validation')
elif ds=='nq_open': data=load_dataset('google-research-datasets/nq_open',split='validation')
elif ds=='tqa': data=load_dataset('truthful_qa','generation')['validation']
N=min(len(data),820)
print("N=",N)
tok=AutoTokenizer.from_pretrained(path); model=AutoModelForCausalLM.from_pretrained(path,torch_dtype=torch.float16,device_map='auto'); model.eval()
nL=model.config.num_hidden_layers; layers=sorted(set([int(nL*f) for f in (0.35,0.5,0.65,0.8,0.9)]))
print('model',model_name,'ds',ds,'nLayers',nL,'probe layers',layers)
ans_dir=f'/root/tsv/save_for_eval/{ds}_hal_det/answers'
feats={L:[] for L in layers}; labels=[]; kept=[]
for i in range(N):
    q=data[i]['question']
    try: a=np.load(f'{ans_dir}/most_likely_hal_det_{model_name}_{ds}_answers_index_{i}.npy')
    except FileNotFoundError: continue
    a=str(a.reshape(-1)[0])
    if ds=='triviaqa': golds=data[i]['answer']['aliases']
    elif ds=='nq_open': golds=data[i]['answer']
    else: golds=[data[i]['best_answer']]+list(data[i]['correct_answers'])
    lab=match(a,golds)
    prompt=f'Answer the question concisely. Q: {q} A:{a}'
    ids=tok(prompt,return_tensors='pt').input_ids.cuda()
    with torch.no_grad(): out=model(ids,output_hidden_states=True)
    for L in layers: feats[L].append(out.hidden_states[L][0,-1,:].float().cpu().numpy())
    labels.append(lab); kept.append(i)
    if i%100==0: print(i,'label_rate',round(np.mean(labels),3))
out={f'X_L{L}':np.stack(feats[L]) for L in layers}; out['labels']=np.array(labels); out['kept']=np.array(kept); out['layers']=np.array(layers)
np.savez(f'/root/tsv/feats_{model_name}_{ds}.npz', **out)
print('SAVED', f'feats_{model_name}_{ds}.npz','n',len(labels),'pos_rate',round(float(np.mean(labels)),3))
