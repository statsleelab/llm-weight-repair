import torch; torch.set_num_threads(1)
import numpy as np, sys, time
from gpt2_lib import *
from detect_lib import *
sd=load_weights("damaged.safetensors")
for k in sys.argv[1:]:
    t=time.time()
    w=sd[k].numpy(); L=llr_map(w)
    v,a,b,l,e=find_block(L)
    frac=(b-a)*(e-l)/w.size
    inside=L[a:b,l:e].mean(); 
    print(k,"block rows %d-%d cols %d-%d frac %.4f score %.0f mean_in %.3f mean_out %.3f  %.1fs"%(a,b,l,e,frac,v,inside,(L.sum()-L[a:b,l:e].sum())/(L.size-(b-a)*(e-l)),time.time()-t))
