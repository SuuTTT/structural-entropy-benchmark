import os,sys,numpy as np,torch
os.environ.setdefault('HF_HOME','/root/.cache/huggingface')
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
N=300; P=3; J='Qwen/Qwen2.5-7B-Instruct'
def qs(ds):
    if ds=='triviaqa': d=load_dataset('trivia_qa','rc.nocontext',split='validation')
    elif ds=='nq': d=load_dataset('google-research-datasets/nq_open',split='validation')
    elif ds=='squad': d=load_dataset('rajpurkar/squad',split='validation')
    elif ds=='sciq': d=load_dataset('allenai/sciq',split='validation')
    elif ds=='tqa': return [x['question'] for x in load_dataset('truthful_qa','generation')['validation'].select(range(N))]
    return [x['question'] for x in d.select(range(min(N,len(d))))]
tok=AutoTokenizer.from_pretrained(J); model=AutoModelForCausalLM.from_pretrained(J,torch_dtype=torch.float16,device_map='auto').eval()
@torch.no_grad()
def para(q):
    msg=[{'role':'user','content':f"Rephrase this question in {P} different ways that keep the EXACT same meaning (just different wording). Output exactly {P} lines, one rephrasing per line, nothing else.\nQuestion: {q}"}]
    ids=tok.apply_chat_template(msg,add_generation_prompt=True,return_tensors='pt').cuda()
    o=model.generate(ids,max_new_tokens=100,do_sample=True,temperature=0.8,top_p=0.95,pad_token_id=tok.eos_token_id)
    t=tok.decode(o[0,ids.shape[1]:],skip_special_tokens=True).strip()
    lines=[l.strip(' -0123456789.').strip() for l in t.split(chr(10)) if l.strip()][:P]
    while len(lines)<P: lines.append(q)
    return lines
for ds in sys.argv[1:]:
    Q=qs(ds); PA=[para(q) for q in Q]
    np.savez(f'/root/uprobe/para_{ds}.npz',question=np.array(Q,object),para=np.array(PA,object))
    print('PARA',ds,'n',len(Q),'ex:',PA[0][0][:50],flush=True)
