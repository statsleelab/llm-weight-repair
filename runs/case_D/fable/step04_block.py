import torch, numpy as np
torch.set_num_threads(1)
from gpt2_lib import *
sd = load_weights("damaged.safetensors")
W = sd["wpe.weight"].numpy()
r0,r1,c0,c1=151,253,59,136
B=W[r0:r1,c0:c1]
print("block shape",B.shape,B.size/W.size)
print("block std",B.std(),"mean",B.mean(),"kurt",((B-B.mean())**4).mean()/B.std()**4)
print("above rows std",W[50:150,c0:c1].std(),"below",W[260:360,c0:c1].std(), "col-neighb", W[r0:r1,140:220].std(), "global",W.std())
# smoothness inside block
print("row-diff std in block",np.diff(B,axis=0).std(),"outside",np.diff(W[260:360,c0:c1],axis=0).std())
print("col-diff std in block",np.diff(B,axis=1).std())
# expected by interpolation between row 150 and 253 (crude)
t=np.linspace(0,1,104)[1:-1,None]
E=W[150,c0:c1][None]*(1-t)+W[253,c0:c1][None]*t
def cc(a,b): return np.corrcoef(a.ravel(),b.ravel())[0,1]
print("corr block vs interp",cc(B,E))
for name,X in [("flipud",B[::-1]),("fliplr",B[:,::-1]),("neg",-B),("rot",B[::-1,::-1])]:
    print(name,cc(X,E))
# sorted values vs expected: permutation?
print("sorted corr",cc(np.sort(B.ravel()),np.sort(E.ravel())), "std E",E.std())
# per-row and per-col corr with E
print("per-col sorted-match: std of B cols vs E cols corr",cc(B.std(0),E.std(0)),"means",cc(B.mean(0),E.mean(0)))
print("row means corr",cc(B.mean(1),E.mean(1)), "row std corr",cc(B.std(1),E.std(1)))
# Is the block copied from elsewhere in W? search best correlation with same-shaped windows
best=[]
from numpy.lib.stride_tricks import sliding_window_view
Bn=(B-B.mean())/B.std()
for rr in range(0,1024-102+1):
    win=sliding_window_view(W[rr:rr+102],77,axis=1)  # (102, 692, 77)
    c=np.einsum('ijk,ik->j',win,Bn)/B.size
    j=np.abs(c).argmax(); best.append((abs(c[j]),rr,j))
best.sort(reverse=True); print(best[:5])
# quantile comparison with gaussian
print("quantiles B/std",np.quantile(B/B.std(),[.001,.01,.1,.5,.9,.99,.999]))
