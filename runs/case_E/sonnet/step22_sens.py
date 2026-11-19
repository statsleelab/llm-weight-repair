from common import *
sd=load_weights("damaged.safetensors")
w=sd["wte.weight"]
z,_,med=zscore(w)
for c in [496,430,36,374,314,442]:
    col=w[:,c]-med[0,c]
    mad=col.abs().median().item()*1.4826
    q=col.abs().sort(descending=True).values
    print(c,"mad",round(mad,4),"std",round(col.std().item(),3),"lnf",round(sd["ln_f.weight"][c].item(),1),"top|dev|",[round(v,2) for v in q[[0,10,50,100,200,400,800,1600,3200]].tolist()])
# distribution of z in a normal column vs corrupt: histogram of |col dev|/mad for col 496
col=(w[:,496]-med[0,496]).abs()/ (( w[:,496]-med[0,496]).abs().median()*1.4826)
for t in [2,3,4,5,6,8,12,20]: print(t,(col>t).sum().item(),end="; ")
