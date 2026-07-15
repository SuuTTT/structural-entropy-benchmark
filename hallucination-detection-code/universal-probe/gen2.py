import os,sys,numpy as np,torch
os.environ.setdefault('HF_HOME','/root/.cache/huggingface'); os.environ.setdefault('HF_HUB_ENABLE_HF_TRANSFER','1')
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
m=sys.argv[1]; ds=sys.argv[2]; N=int(sys.argv[3]) if len(sys.argv)>3 else 300; K=8; P=3
HF={'qwen14b':'Qwen/Qwen2.5-14B','qwen14bi':'Qwen/Qwen2.5-14B-Instruct','mistral7b':'mistralai/Mistral-7B-v0.1','mistral7bi':'mistralai/Mistral-7B-Instruct-v0.2','yi9b':'01-ai/Yi-1.5-9B-Chat','qwen7b':'Qwen/Qwen2.5-7B','qwen7bi':'Qwen/Qwen2.5-7B-Instruct','llama8b':'NousResearch/Meta-Llama-3.1-8B','llama8bi':'NousResearch/Meta-Llama-3.1-8B-Instruct'}
def load_ds(ds):
    if ds=='triviaqa': d=load_dataset('trivia_qa','rc.nocontext',split='validation'); return [(x['question'],list(x['answer']['aliases'])) for x in d.select(range(min(N,len(d))))]
    if ds=='nq': d=load_dataset('google-research-datasets/nq_open',split='validation'); return [(x['question'],list(x['answer'])) for x in d.select(range(min(N,len(d))))]
    if ds=='squad': d=load_dataset('rajpurkar/squad',split='validation'); return [(x['question'],list(x['answers']['text'])) for x in d.select(range(min(N,len(d))))]
    if ds=='sciq': d=load_dataset('allenai/sciq',split='validation'); return [(x['question'],[x['correct_answer']]) for x in d.select(range(min(N,len(d))))]
    if ds=='tqa': d=load_dataset('truthful_qa','generation')['validation']; return [(x['question'],[x['best_answer']]+list(x['correct_answers'])) for x in d.select(range(min(N,len(d))))]
data=load_ds(ds); path=HF[m]
PARA=np.load('/root/uprobe/para_'+ds+'.npz',allow_pickle=True)['para']
tok=AutoTokenizer.from_pretrained(path); tok.pad_token=tok.pad_token or tok.eos_token
model=AutoModelForCausalLM.from_pretrained(path,torch_dtype=torch.float16,device_map='auto').eval()
nL=model.config.num_hidden_layers; layers=sorted(set(int(nL*f) for f in (0.4,0.55,0.7,0.85)))
print(m,ds,'nL',nL,'layers',layers,'N',len(data),flush=True)
def gen(prompt,**kw):
    ids=tok(prompt,return_tensors='pt').input_ids.cuda()
    with torch.no_grad(): o=model.generate(ids,pad_token_id=tok.eos_token_id,**kw)
    return [tok.decode(o[j,ids.shape[1]:],skip_special_tokens=True).strip() for j in range(o.shape[0])], ids
feat={L:[] for L in layers}; G=[];SAMP=[];PARAQ=[];PARAANS=[];GOLD=[];Q=[]
for i,(q,golds) in enumerate(data):
    ap=f'Answer the question concisely. Q: {q} A:'
    greedy,ids=gen(ap,max_new_tokens=48,do_sample=False,num_beams=1)
    greedy=greedy[0]
    samps,_=gen(ap,max_new_tokens=48,do_sample=True,temperature=0.7,top_p=0.9,num_return_sequences=K)
    paras=list(PARA[i])
    paraans=[]
    for pq in paras:
        pq=str(pq).strip() or q
        a,_=gen(f'Answer the question concisely. Q: {pq} A:',max_new_tokens=48,do_sample=False,num_beams=1); paraans.append(a[0])
    fp=tok(f'{ap}{greedy}',return_tensors='pt').input_ids.cuda()
    with torch.no_grad(): out=model(fp,output_hidden_states=True)
    for L in layers: feat[L].append(out.hidden_states[L][0,-1,:].float().cpu().numpy())
    G.append(greedy);SAMP.append(samps);PARAQ.append(paras);PARAANS.append(paraans);GOLD.append(list(golds));Q.append(q)
    if i%50==0: print(i,flush=True)
o={f'X_L{L}':np.stack(feat[L]) for L in layers}
o.update(greedy=np.array(G,object),samples=np.array(SAMP,object),para_q=np.array(PARAQ,object),para_ans=np.array(PARAANS,object),gold=np.array(GOLD,object),question=np.array(Q,object),layers=np.array(layers))
np.savez(f'/root/uprobe/raw2/{m}_{ds}.npz',**o)
print('SAVED',m,ds,'n',len(G),flush=True)
