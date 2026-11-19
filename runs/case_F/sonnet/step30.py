from repairlib import *
import torch.nn.functional as F
s2=build2(hard=0.5)
ids=devids(300)
x=torch.tensor([[EOT]+ids[:200]])
B,T=x.shape
h=s2['wte.weight'][x]+s2['wpe.weight'][:T]
mask=torch.tril(torch.ones(T,T,dtype=torch.bool))
pos=100
for i in range(12):
    p=f"h.{i}."
    ln=F.layer_norm(h,(768,),s2[p+"ln_1.weight"],s2[p+"ln_1.bias"],1e-5)
    qkv=ln@s2[p+"attn.c_attn.weight"]+s2[p+"attn.c_attn.bias"]
    q,k,v=qkv.split(768,-1)
    sh=lambda t:t.view(B,T,12,64).transpose(1,2)
    q,k,v=sh(q),sh(k),sh(v)
    att=((q@k.transpose(-1,-2))/8).masked_fill(~mask,float('-inf')).softmax(-1)
    a=(att@v).transpose(1,2).reshape(B,T,768)
    ao=a@s2[p+"attn.c_proj.weight"]+s2[p+"attn.c_proj.bias"]
    h=h+ao
    ln2=F.layer_norm(h,(768,),s2[p+"ln_2.weight"],s2[p+"ln_2.bias"],1e-5)
    mo=gelu_new(ln2@s2[p+"mlp.c_fc.weight"]+s2[p+"mlp.c_fc.bias"])@s2[p+"mlp.c_proj.weight"]+s2[p+"mlp.c_proj.bias"]
    h=h+mo
    print(i,"attn out rms %.1f max %.1f | mlp out rms %.1f max %.1f(dim %d) | h rms %.1f | att to pos0 %.2f"%(ao[0,pos].pow(2).mean().sqrt(),ao[0,pos].abs().max(),mo[0,pos].pow(2).mean().sqrt(),mo[0,pos].abs().max(),mo[0,pos].abs().argmax(),h[0,pos].pow(2).mean().sqrt(),att[0,:,pos,0].mean()))
