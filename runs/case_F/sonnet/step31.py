from repairlib import *
ids=devids(2048)
for split in [4,8]:
    s2=build3(split=split)
    print(split,perplexity(s2,ids,window=512),flush=True)
