import numpy as np
def impute(X, M, lam=0.1, iters=3, center=True):
    """X: (n,d) samples x dims; M: bool mask of missing. Returns completed X.
    Gaussian conditional-mean imputation with EM-style covariance and shrinkage lam toward diag."""
    X=X.astype(np.float64).copy(); n,d=X.shape
    obs=~M
    cnt=obs.sum(0)
    mu=(X*obs).sum(0)/cnt if center else np.zeros(d)
    X[M]=np.broadcast_to(mu,X.shape)[M]
    rows=np.where(M.any(1))[0]
    for it in range(iters):
        mu=X.mean(0) if center else np.zeros(d)
        R=X-mu
        S=R.T@R/n
        dg=np.diag(S)
        S=(1-lam)*S+lam*np.diag(dg)
        P=np.linalg.inv(S)
        Xn=X.copy()
        for i in rows:
            m=np.where(M[i])[0]
            r=X[i]-mu; r[m]=0
            g=P@r
            Xn[i,m]=mu[m]-np.linalg.solve(P[np.ix_(m,m)],g[m])
        X=Xn
    return X
