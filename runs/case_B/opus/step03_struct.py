import torch; torch.set_num_threads(1)
from gpt2_lib import *
sd = load_weights("damaged.safetensors")
def rmad(x,dim): return (x.abs().median(dim=dim,keepdim=True).values*1.4826)
for k in ["h.0.attn.c_attn.weight","h.2.mlp.c_proj.weight","h.3.mlp.c_proj.weight","h.11.attn.c_proj.weight","wpe.weight","wte.weight"]:
    v=sd[k]; s=rmad(v.flatten(),0)
    m=(v.abs()>8*s)
    rc=m.sum(1).float(); cc=m.sum(0).float()
    print(k, "n",m.sum().item(),"rows: mean %.2f max %d argmax %d | cols: mean %.2f max %d argmax %d"%(rc.mean(),rc.max(),rc.argmax(),cc.mean(),cc.max(),cc.argmax()))
    # top 8 entries
    idx=v.abs().flatten().topk(8).indices
    print("   top entries", [(int(i//v.shape[1]),int(i%v.shape[1]),round(v.flatten()[i].item(),2)) for i in idx])
    # column scale variation
    cs=rmad(v,0).flatten(); rs=rmad(v,1).flatten()
    print("   col robust sd range %.3f-%.3f, row robust sd range %.3f-%.3f"%(cs.min(),cs.max(),rs.min(),rs.max()))
