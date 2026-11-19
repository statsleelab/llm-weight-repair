import numpy as np
def llr_map(w):
    w=w.astype(np.float64); R,C=w.shape
    sigma=w.std()
    s=w**2+1e-12
    rs=np.ones(R); cs=np.ones(C)
    for _ in range(4):
        rs=np.median(s/cs[None],1); cs=np.median(s/rs[:,None],0)
    sc=np.sqrt(rs[:,None]*cs[None])
    n=w/sc
    lim=np.quantile(np.abs(n),0.9999)*1.2
    bins=np.linspace(-lim,lim,401)
    h,_=np.histogram(np.clip(n,-lim,lim*0.9999),bins=bins,density=True)
    from numpy import convolve
    h=np.convolve(h,np.ones(5)/5,mode='same')+1e-6
    idx=np.clip(np.digitize(n,bins)-1,0,399)
    lp_orig=np.log(h[idx])-np.log(sc)
    lp_noise=-0.5*(w/sigma)**2-np.log(sigma)-0.5*np.log(2*np.pi)
    return np.clip(lp_noise-lp_orig,-20,20)

def max_rect(S):
    # max-sum subrectangle; S small (r x c)
    r,c=S.shape; best=(-1e18,0,0,0,0)
    cum=np.vstack([np.zeros((1,c)),np.cumsum(S,0)])
    for t in range(r):
        rowsums=cum[t+1:]-cum[t]  # (r-t, c) sums rows t..b
        # kadane per bottom vectorized: max subarray of each row
        pc=np.hstack([np.zeros((rowsums.shape[0],1)),np.cumsum(rowsums,1)])
        runmin=np.minimum.accumulate(pc,1)
        val=pc[:,1:]-runmin[:,:-1]
        i=np.unravel_index(int(val.argmax()),val.shape)
        if val[i]>best[0]:
            b,e=i; l=int(np.argmin(pc[b,:e+1]))
            best=(val[i],t,t+b+1,l,e+1)
    return best

def find_block(L,tile=8):
    R,C=L.shape; r,c=R//tile,C//tile
    T=L[:r*tile,:c*tile].reshape(r,tile,c,tile).sum((1,3))
    v,a,b,l,e=max_rect(T)
    a,b,l,e=a*tile,b*tile,l*tile,e*tile
    # refine at full res
    cr=np.vstack([np.zeros((1,C)),np.cumsum(L,0)])
    def tot(a,b,l,e): return L[a:b,l:e].sum()
    for _ in range(4):
        cand=[(tot(x,b,l,e),x) for x in range(max(0,a-tile),min(b,a+tile)+1)]; a=max(cand)[1]
        cand=[(tot(a,x,l,e),x) for x in range(max(a+1,b-tile),min(R,b+tile)+1)]; b=max(cand)[1]
        cand=[(tot(a,b,x,e),x) for x in range(max(0,l-tile),min(e,l+tile)+1)]; l=max(cand)[1]
        cand=[(tot(a,b,l,x),x) for x in range(max(l+1,e-tile),min(C,e+tile)+1)]; e=max(cand)[1]
    return tot(a,b,l,e),a,b,l,e
