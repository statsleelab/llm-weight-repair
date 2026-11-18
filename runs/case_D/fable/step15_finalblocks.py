import json
b=json.load(open("blocks.json"))   # E2 fixed-size argmax (step07)
shapes={k:(v[1]-v[0],v[3]-v[2]) for k,v in b.items()}
over={"h.10.attn.c_attn.weight":(622,1851),"h.11.attn.c_proj.weight":(375,514),"h.2.attn.c_attn.weight":(458,65),
      "h.10.attn.c_proj.weight":(419,238),"h.9.attn.c_attn.weight":(144,1648),"h.8.attn.c_attn.weight":(43,1732),"wpe.weight":(151,59)}
for k,(i,j) in over.items():
    h,w=shapes[k]; b[k]=[i,i+h,j,j+w]
json.dump(b,open("blocks_final.json","w"),indent=0)
for k,v in b.items(): print(k,v)
