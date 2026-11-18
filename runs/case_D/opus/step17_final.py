import torch
torch.set_num_threads(1)
src=open("step16_fill.py").read(); exec(src[:src.index("# held-out")])  # reuse load + fill functions from step16
N=8192
print("damaged", perplexity(sd0,ids,max_tokens=N),flush=True)
best=None
for al in [0.1,1.0]:
    sd=build("ridge",al); p=perplexity(sd,ids,max_tokens=N); print("ridge alpha",al,p,flush=True)
    if best is None or p<best[0]: best=(p,al,sd)
save_weights(best[2],"repaired.safetensors"); print("saved alpha",best[1])
print(generate(best[2],tok,"The capital of France is",15))
