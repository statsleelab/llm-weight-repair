from step20_mixed import *
for T,kc in [(20,5),(30,5),(45,5)]:
    r,n=build(T,kc)
    _,h=forward(r,torch.tensor([[EOT]+ids[:40]]),return_hidden=True)
    print(T,kc,n,nll_windows(r,ids,window=256),"pos0 max l3",round(h[3][0,0].abs().max().item()),flush=True)
