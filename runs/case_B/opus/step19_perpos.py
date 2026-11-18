import torch; torch.set_num_threads(1)
import torch.nn.functional as F
from gpt2_lib import *
from common import *
D = load_weights("damaged.safetensors")
tok=Tokenizer("tokenizer"); ids=tok.encode(open("dev.txt").read())
cache={k:detect2(v) for k,v in D.items() if v.dim()==2 and k!="wpe.weight"}
sd=dict(D)
for k in cache:
    v=D[k]; z,mu,s=cache[k]; r=v-mu; tau=r[z.abs()>15].std().item()
    m=(z.abs()>5)&(r.abs()<5*tau); o=v.clone(); o[m]=mu[m]; sd[k]=o
zw,muw,sw=detect2(D["wpe.weight"]); w=D["wpe.weight"].clone(); m=zw.abs()>5; w[m]=muw[m]; sd["wpe.weight"]=w
torch.save(sd,"/tmp/claude-0/-home-user/19958a03-350d-552e-9865-2b580b11f60f/scratchpad/r5g.pt")
x=torch.tensor([[EOT]+ids[:512],[EOT]+ids[512:1024]])
lg=forward(sd,x[:,:-1]); nll=F.cross_entropy(lg.reshape(-1,VOCAB),x[:,1:].reshape(-1),reduction='none').view(2,-1)
for a,b in [(0,4),(4,16),(16,64),(64,256),(256,512)]: print(f"pos {a}-{b}: nll {nll[:,a:b].mean():.2f}")
print("top1 acc", (lg.argmax(-1)==x[:,1:]).float().mean().item())
print("pred sample:", tok.decode(lg[0,:40].argmax(-1).tolist()))
print("true      :", tok.decode(x[0,1:41].tolist()))
