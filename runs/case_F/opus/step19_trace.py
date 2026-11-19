import torch, math
torch.set_num_threads(1)
import torch.nn.functional as F
from gpt2_lib import *
import repair3 as R
ids=torch.load("dev_ids.pt").tolist()
sd=load_weights("damaged.safetensors")
r=R.repair(sd,mode="soft",nu=4,line_adapt=True)
torch.save(r,"cand_soft.pt")
x=torch.tensor([[EOT]+ids[:63]])
B,T=x.shape
h=r["wte.weight"][x]+r["wpe.weight"][:T]
mask=torch.tril(torch.ones(T,T,dtype=torch.bool))
for i in range(4):
    p=f"h.{i}."
    a_in=F.layer_norm(h,(768,),r[p+"ln_1.weight"],r[p+"ln_1.bias"],1e-5)
    qkv=a_in@r[p+"attn.c_attn.weight"]+r[p+"attn.c_attn.bias"]
    q,k,v=qkv.split(768,-1)
    q=q.view(B,T,12,64).transpose(1,2);k=k.view(B,T,12,64).transpose(1,2);v=v.view(B,T,12,64).transpose(1,2)
    att=((q@k.transpose(-1,-2))/8).masked_fill(~mask,float('-inf')).softmax(-1)
    a=(att@v).transpose(1,2).reshape(B,T,768)@r[p+"attn.c_proj.weight"]+r[p+"attn.c_proj.bias"]
    h=h+a
    m_in=F.layer_norm(h,(768,),r[p+"ln_2.weight"],r[p+"ln_2.bias"],1e-5)
    pre=m_in@r[p+"mlp.c_fc.weight"]+r[p+"mlp.c_fc.bias"]
    act=gelu_new(pre)
    m=act@r[p+"mlp.c_proj.weight"]+r[p+"mlp.c_proj.bias"]
    h=h+m
    print(i,"attn447 pos0 %.1f pos1+ %.1f | mlp447 pos0 %.1f pos1+ %.1f"%(a[0,0,447],a[0,1:,447].mean(),m[0,0,447],m[0,1:,447].mean()))
    # top neurons contributing to dim 447 at pos1+
    contrib=(act[0,1:]*r[p+"mlp.c_proj.weight"][:,447]).mean(0)
    t=contrib.abs().topk(4).indices
    print("   top neurons",[(int(n),round(contrib[n].item(),1),round(act[0,0,n].item(),2),round(act[0,1:,n].mean().item(),2),round(r[p+"mlp.c_proj.weight"][n,447].item(),2)) for n in t])
