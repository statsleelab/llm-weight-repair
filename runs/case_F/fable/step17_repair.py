# v6: as v5 (replib) but finer z-bins (replib2).  usage: step17_repair.py OUT KEEP A PART(nowte|wte)
import torch, sys, os
torch.set_num_threads(1)
import replib2 as R
from gpt2_lib import load_weights, save_weights
OUT=sys.argv[1]; R.KEEP=float(sys.argv[2]); R.A=float(sys.argv[3]); PART=sys.argv[4]
sd=load_weights("damaged.safetensors")
out=load_weights(OUT) if os.path.exists(OUT) else dict(sd)
for k,v in sd.items():
    if v.ndim!=2 or (k=="wte.weight")!=(PART=="wte"): continue
    out[k]=R.repair_matrix(k,v)
save_weights(out,OUT)
