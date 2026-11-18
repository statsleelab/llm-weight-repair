import torch,numpy as np
torch.set_num_threads(1)
from gpt2_lib import *
import step15_soft as S
from step30_sink import sink
d=S.sd['wpe.weight']
cs=d[1:].abs().median(0).values; r0=d[0]; 
big=(r0.abs()>0.25)&(cs<0.02)
print("row0 suspicious entries",big.sum().item(),"| row0 |w| >0.25 total",(r0.abs()>0.25).sum().item(),"| cols scale of those",cs[big][:10])
print("row1 vals at those",d[1][big][:10],"row0",r0[big][:10])
c=load_weights("cand_heavy.safetensors")
w0=c['wpe.weight'].clone(); w0[0]=torch.where(big,d[1],w0[0])   # replace with row1 value (positions 0/1 are close in non-special columns)
c2=dict(c); c2['wpe.weight']=w0
print("sink before",sink(c),"after",sink(c2))
ids=S.ids[:1100]
print("ppl after %.4g"%perplexity(c2,ids,max_tokens=1024))
save_weights(c2,"cand_row0.safetensors")
