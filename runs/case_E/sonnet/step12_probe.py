from common import *
import torch.nn.functional as F
sd=load_weights("damaged.safetensors")
tok=Tokenizer("tokenizer")
ids=tok.encode(open("dev.txt").read())[:100]
x=torch.tensor([[EOT]+ids]); T=x.shape[1]
e=sd["wte.weight"][x]+sd["wpe.weight"][:T]
p="h.0."
h=F.layer_norm(e,(768,),sd[p+"ln_1.weight"],sd[p+"ln_1.bias"],1e-5)
qkv=h@sd[p+"attn.c_attn.weight"]+sd[p+"attn.c_attn.bias"]
q,k,v=qkv.split(768,-1)
q=q.view(1,T,12,-1).transpose(1,2);k=k.view(1,T,12,-1).transpose(1,2);v=v.view(1,T,12,-1).transpose(1,2)
att=(q@k.transpose(-1,-2))/8; mask=torch.tril(torch.ones(T,T,dtype=torch.bool))
att=att.masked_fill(~mask,float("-inf")).softmax(-1)
a=(att@v).transpose(1,2).reshape(1,T,768)
x1=e+a@sd[p+"attn.c_proj.weight"]+sd[p+"attn.c_proj.bias"]
print("after attn: rms",x1.pow(2).mean().sqrt().item(),"pos0 447",x1[0,0,447].item(),"pos5 447",x1[0,5,447].item(),"pos5 absmax",x1[0,5].abs().max().item(),x1[0,5].abs().argmax().item())
h2=F.layer_norm(x1,(768,),sd[p+"ln_2.weight"],sd[p+"ln_2.bias"],1e-5)
pre=h2@sd[p+"mlp.c_fc.weight"]+sd[p+"mlp.c_fc.bias"]
m=gelu_new(pre)
print("mlp act absmax per pos",m.abs().amax(-1)[0,:8], "argmax", m.abs().argmax(-1)[0,:8])
print("neuron 2710 act pos0..5",m[0,:6,2710])
print("preact absmax",pre.abs().max().item())
out=m@sd[p+"mlp.c_proj.weight"]+sd[p+"mlp.c_proj.bias"]
print("mlp out rms",out.pow(2).mean().sqrt().item(),"447 pos5",out[0,5,447].item(),"pos0",out[0,0,447].item())
# top contributions to out[5,447]
c=m[0,5]*sd[p+"mlp.c_proj.weight"][:,447]
print(c.abs().topk(5))
