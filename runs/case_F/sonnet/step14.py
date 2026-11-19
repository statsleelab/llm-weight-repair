from repairlib import *
ids=devids(2048)
for t,am in [(3,0.2),(5,0.5),(8,0.8)]:
    s2=build(t,am)
    print(t,am,perplexity(s2,ids,window=512),flush=True)
