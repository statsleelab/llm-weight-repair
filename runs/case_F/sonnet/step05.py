from common import *
sd=load_weights("damaged.safetensors")
for k,v in sd.items():
    if v.dim()==2:
        a=v.abs().flatten()
        big=(a>0.6).float().mean().item()
        # fraction in 0.6-3 vs overall; also global MAD scale
        mad=(v-v.median()).abs().median().item()*1.4826
        print(f"{k:28s} mad {mad:.3f} std {v.std():.3f} >0.6:{big:.4f} >3:{(a>3).float().mean():.5f} max {a.max():.1f}")
