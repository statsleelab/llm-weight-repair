from common import *
sd=load_weights("damaged.safetensors")
w=sd['wpe.weight']
# median of 4 neighbors (±1,±2) excluding self
pad=torch.nn.functional.pad(w.T[None],(2,2),mode='replicate')[0].T
nb=torch.stack([pad[0:-4],pad[1:-3],pad[3:-1],pad[4:]],0)
med=nb.median(0).values
r=(w-med).abs()
print(torch.quantile(r.flatten()[:800000],torch.tensor([.5,.9,.95,.96,.97,.98,.99])))
for t in [0.02,0.05,0.1,0.2,0.4]: print(t,(r>t).float().mean().item())
