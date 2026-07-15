import os,sys,glob,numpy as np,torch
os.environ.setdefault('HF_HOME','/root/.cache/huggingface')
from transformers import AutoTokenizer, AutoModelForCausalLM
J='Qwen/Qwen2.5-7B-Instruct'
tok=AutoTokenizer.from_pretrained(J); model=AutoModelForCausalLM.from_pretrained(J,torch_dtype=torch.float16,device_map='auto').eval()
@torch.no_grad()
def judge(q,gold,ans):
    msg=[{'role':'user','content':f"Grade this QA answer. Question: {q}\nAcceptable reference answers: {'; '.join([str(g) for g in gold[:5]])}\nStudent answer: {ans}\nDoes the student answer match any reference in meaning? Reply ONLY 'yes' or 'no'."}]
    ids=tok.apply_chat_template(msg,add_generation_prompt=True,return_tensors='pt').cuda()
    o=model.generate(ids,max_new_tokens=3,do_sample=False,pad_token_id=tok.eos_token_id)
    t=tok.decode(o[0,ids.shape[1]:],skip_special_tokens=True).strip().lower()
    return 1 if t.startswith('y') else 0
for f in sorted(glob.glob('/root/uprobe/raw2/*.npz')):
    nm=f.split('/')[-1].replace('.npz',''); out=f'/root/uprobe/feats2/judge_{nm}.npy'
    if os.path.exists(out): continue
    d=np.load(f,allow_pickle=True); Q=d['question'];G=d['gold'];GR=d['greedy']
    lab=np.array([judge(Q[i],list(G[i]),str(GR[i])) for i in range(len(Q))])
    np.save(out,lab); print('JUDGED',nm,'n',len(lab),'pos',round(float(lab.mean()),3),flush=True)
