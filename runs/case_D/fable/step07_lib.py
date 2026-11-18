import torch
def loo2(W, lamf):
    W=W.double(); n,m=W.shape; out=[]
    G=W@W.T/m; Th=torch.linalg.inv(G+lamf*G.diagonal().mean()*torch.eye(n,dtype=W.dtype)); out.append(((Th@W)/Th.diagonal()[:,None]).numpy())
    G=W.T@W/n; Th=torch.linalg.inv(G+lamf*G.diagonal().mean()*torch.eye(m,dtype=W.dtype)); out.append(((W@Th)/Th.diagonal()[None,:]).numpy())
    return out
