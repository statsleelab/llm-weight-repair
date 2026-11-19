import json
b=json.load(open("blocks_comb.json"))
b["h.10.attn.c_proj.weight"]=[603,736,463,596]
b["h.11.attn.c_proj.weight"]=[487,620,138,271]
b["wpe.weight"]=[125,302,591,724]
json.dump(b,open("blocks_final.json","w"),indent=1)
