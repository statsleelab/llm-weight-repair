from repairlib import *
import torch.nn.functional as F
s2=build(4)
x=torch.tensor([[EOT]+devids(30)])
B,T=x.shape
h=s2['wte.weight'][x]+s2['wpe.weight'][:T]
mask=torch.tril(torch.ones(T,T,dtype=torch.bool))
for i in range(3):
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
    print(i,"attn out dim138 pos0,1,2:",[round(ao[0,j,138].item(),1) for j in range(3)],"dim447",[round(ao[0,j,447].item(),1) for j in range(3)])
    ln2=F.layer_norm(h,(768,),s2[p+"ln_2.weight"],s2[p+"ln_2.bias"],1e-5)
    pre=ln2@s2[p+"mlp.c_fc.weight"]+s2[p+"mlp.c_fc.bias"]
    m=gelu_new(pre)
    mo=m@s2[p+"mlp.c_proj.weight"]+s2[p+"mlp.c_proj.bias"]
    contrib=m[0]*s2[p+"mlp.c_proj.weight"][:,138]
    for pos in [0,1,5]:
        top=contrib[pos].abs().topk(3).indices
        print("   pos",pos,"mlp dim138 out %.1f"%mo[0,pos,138].item(),[(j.item(),round(pre[0,pos,j].item(),1),round(contrib[pos,j].item(),1)) for j in top])
    h=h+mo
