from repairlib import *
w=sd['wpe.weight']
torch.set_printoptions(precision=2,linewidth=200)
print(w[:8][:,[138,378,447,373,64]])
print(w[8:14][:,[138,378,447,373,64]])
print("late rows end",w[-6:][:,[138,378,447]])
